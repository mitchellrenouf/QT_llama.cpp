use mrml_error::{Result, anyhow};
use mrml_runtime::Vector;

#[derive(Clone)]
enum Atom {
    Literal(char),
    Any,
    Class { chars: Vector<char>, negated: bool },
}

#[derive(Clone, Copy)]
enum Repeat {
    One,
    ZeroOrOne,
    ZeroOrMore,
    OneOrMore,
}

#[derive(Clone)]
struct Piece {
    atom: Atom,
    repeat: Repeat,
}

pub struct Regex {
    pieces: Vector<Piece>,
    anchored_start: bool,
    anchored_end: bool,
}

impl Regex {
    pub fn new(pattern: &str) -> Result<Self> {
        let mut chars = pattern.chars().peekable();
        let anchored_start = matches!(chars.peek(), Some('^'));
        if anchored_start {
            chars.next();
        }
        let mut pieces = Vector::new();
        let mut anchored_end = false;
        while let Some(ch) = chars.next() {
            if ch == '$' && chars.peek().is_none() {
                anchored_end = true;
                break;
            }
            let atom = match ch {
                '.' => Atom::Any,
                '\\' => Atom::Literal(
                    chars
                        .next()
                        .ok_or_else(|| anyhow!("trailing regex escape"))?,
                ),
                '[' => {
                    let negated = matches!(chars.peek(), Some('^'));
                    if negated {
                        chars.next();
                    }
                    let mut class = Vector::new();
                    let mut closed = false;
                    while let Some(item) = chars.next() {
                        if item == ']' {
                            closed = true;
                            break;
                        }
                        let value = if item == '\\' {
                            chars
                                .next()
                                .ok_or_else(|| anyhow!("trailing class escape"))?
                        } else {
                            item
                        };
                        if matches!(chars.peek(), Some('-')) {
                            chars.next();
                            let end = chars
                                .next()
                                .ok_or_else(|| anyhow!("unfinished character range"))?;
                            if value > end {
                                return Err(anyhow!("reversed character range"));
                            }
                            class.extend(value..=end);
                        } else {
                            class.push(value);
                        }
                    }
                    if !closed {
                        return Err(anyhow!("unclosed character class"));
                    }
                    Atom::Class {
                        chars: class,
                        negated,
                    }
                }
                '*' | '+' | '?' => return Err(anyhow!("regex quantifier has no preceding atom")),
                other => Atom::Literal(other),
            };
            let repeat = match chars.peek() {
                Some('*') => {
                    chars.next();
                    Repeat::ZeroOrMore
                }
                Some('+') => {
                    chars.next();
                    Repeat::OneOrMore
                }
                Some('?') => {
                    chars.next();
                    Repeat::ZeroOrOne
                }
                _ => Repeat::One,
            };
            pieces.push(Piece { atom, repeat });
        }
        Ok(Self {
            pieces,
            anchored_start,
            anchored_end,
        })
    }

    pub fn is_match(&self, text: &str) -> bool {
        let chars: Vector<char> = text.chars().collect();
        if self.anchored_start {
            self.matches_from(&chars, 0, 0)
        } else {
            (0..=chars.len()).any(|start| self.matches_from(&chars, 0, start))
        }
    }

    fn matches_from(&self, text: &[char], piece: usize, position: usize) -> bool {
        if piece == self.pieces.len() {
            return !self.anchored_end || position == text.len();
        }
        let current = &self.pieces[piece];
        let accepts = |index: usize| index < text.len() && atom_matches(&current.atom, text[index]);
        match current.repeat {
            Repeat::One => accepts(position) && self.matches_from(text, piece + 1, position + 1),
            Repeat::ZeroOrOne => {
                self.matches_from(text, piece + 1, position)
                    || (accepts(position) && self.matches_from(text, piece + 1, position + 1))
            }
            Repeat::ZeroOrMore | Repeat::OneOrMore => {
                let minimum = usize::from(matches!(current.repeat, Repeat::OneOrMore));
                let mut end = position;
                while accepts(end) {
                    end += 1;
                }
                if end - position < minimum {
                    return false;
                }
                (position + minimum..=end)
                    .rev()
                    .any(|next| self.matches_from(text, piece + 1, next))
            }
        }
    }
}

fn atom_matches(atom: &Atom, value: char) -> bool {
    match atom {
        Atom::Literal(expected) => *expected == value,
        Atom::Any => true,
        Atom::Class { chars, negated } => chars.contains(&value) != *negated,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supports_search_anchors_classes_and_quantifiers() {
        assert!(Regex::new("g.mm.").unwrap().is_match("gamma"));
        assert!(Regex::new("^ab+c$").unwrap().is_match("abbbc"));
        assert!(Regex::new("[a-z]+[0-9]?").unwrap().is_match("item7"));
        assert!(!Regex::new("^ab+c$").unwrap().is_match("zabbbc"));
    }
}
