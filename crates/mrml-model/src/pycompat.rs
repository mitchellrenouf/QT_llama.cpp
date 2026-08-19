use minijinja::value::{from_args, ValueKind};
use minijinja::{Error, ErrorKind, State, Value};

pub fn unknown_method_callback(
    _state: &State,
    value: &Value,
    method: &str,
    args: &[Value],
) -> Result<Value, Error> {
    match value.kind() {
        ValueKind::String => string_method(value, method, args),
        ValueKind::Map => map_method(value, method, args),
        ValueKind::Seq => sequence_method(value, method, args),
        _ => Err(Error::from(ErrorKind::UnknownMethod)),
    }
}

fn string_method(value: &Value, method: &str, args: &[Value]) -> Result<Value, Error> {
    let text = value.as_str().ok_or_else(|| Error::from(ErrorKind::UnknownMethod))?;
    match method {
        "lower" => { let () = from_args(args)?; Ok(Value::from(text.to_lowercase())) }
        "upper" => { let () = from_args(args)?; Ok(Value::from(text.to_uppercase())) }
        "strip" => { let () = from_args(args)?; Ok(Value::from(text.trim())) }
        "lstrip" => { let () = from_args(args)?; Ok(Value::from(text.trim_start())) }
        "rstrip" => { let () = from_args(args)?; Ok(Value::from(text.trim_end())) }
        "startswith" => {
            let (prefix,): (&str,) = from_args(args)?;
            Ok(Value::from(text.starts_with(prefix)))
        }
        "endswith" => {
            let (suffix,): (&str,) = from_args(args)?;
            Ok(Value::from(text.ends_with(suffix)))
        }
        "replace" => {
            let (old, new, count): (&str, &str, Option<usize>) = from_args(args)?;
            Ok(Value::from(match count { Some(count) => text.replacen(old, new, count), None => text.replace(old, new) }))
        }
        "split" => {
            let (separator,): (Option<&str>,) = from_args(args)?;
            let parts: Vec<Value> = match separator {
                Some(separator) => text.split(separator).map(Value::from).collect(),
                None => text.split_whitespace().map(Value::from).collect(),
            };
            Ok(Value::from(parts))
        }
        "join" => {
            let (values,): (&Value,) = from_args(args)?;
            let parts: Result<Vec<String>, Error> = values.try_iter()?.map(|item| Ok(item.to_string())).collect();
            Ok(Value::from(parts?.join(text)))
        }
        _ => Err(Error::from(ErrorKind::UnknownMethod)),
    }
}

fn map_method(value: &Value, method: &str, args: &[Value]) -> Result<Value, Error> {
    let object = value.as_object().ok_or_else(|| Error::from(ErrorKind::UnknownMethod))?;
    match method {
        "get" => {
            let (key, default): (&Value, Option<Value>) = from_args(args)?;
            Ok(object.get_value(key).or(default).unwrap_or_else(|| Value::from(())))
        }
        "keys" => {
            let () = from_args(args)?;
            Ok(object.try_iter().map(|items| items.collect::<Vec<_>>()).unwrap_or_default().into())
        }
        "values" => {
            let () = from_args(args)?;
            Ok(object.try_iter_pairs().map(|items| items.map(|(_, value)| value).collect::<Vec<_>>()).unwrap_or_default().into())
        }
        "items" => {
            let () = from_args(args)?;
            Ok(object.try_iter_pairs().map(|items| items.map(|(key, value)| Value::from(vec![key, value])).collect::<Vec<_>>()).unwrap_or_default().into())
        }
        _ => Err(Error::from(ErrorKind::UnknownMethod)),
    }
}

fn sequence_method(value: &Value, method: &str, args: &[Value]) -> Result<Value, Error> {
    match method {
        "count" => {
            let (needle,): (&Value,) = from_args(args)?;
            Ok(Value::from(value.try_iter()?.filter(|item| item == needle).count()))
        }
        _ => Err(Error::from(ErrorKind::UnknownMethod)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use minijinja::{context, Environment};

    #[test]
    fn supports_methods_used_by_chat_templates() {
        let mut environment = Environment::new();
        environment.set_unknown_method_callback(unknown_method_callback);
        let template = environment.template_from_str(
            "{{ role.strip().upper() }}|{{ tool.get('name', 'missing') }}|{% for key, value in tool.items() %}{{ key }}={{ value }}{% endfor %}",
        ).unwrap();
        let rendered = template
            .render(context! { role => " assistant ", tool => context! { name => "clock" } })
            .unwrap();
        assert_eq!(rendered, "ASSISTANT|clock|name=clock");
    }
}
