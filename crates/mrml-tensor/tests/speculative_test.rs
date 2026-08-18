use mrml_tensor::speculative::SpeculativeDecoder;

#[test]
fn test_speculative_decoding_pipeline() {
    let mut spec = SpeculativeDecoder::new(3, 4);

    // Learn pattern
    let training_seq = vec![101, 202, 303, 404, 505, 606];
    spec.record_sequence(&training_seq);

    // Context that matches the trained prefix
    let ctx = vec![101, 202, 303];
    let draft = spec.propose_draft_tokens(&ctx);

    assert_eq!(draft.len(), 3);
    assert_eq!(draft[0], 404);
    assert_eq!(draft[1], 505);
    assert_eq!(draft[2], 606);

    // Verification
    let actual = vec![404, 505, 999]; // first 2 match, 3rd differs
    let accepted = spec.verify_draft(&draft, &actual);

    assert_eq!(accepted, 2);
    assert!(spec.acceptance_rate() > 0.0);
}
