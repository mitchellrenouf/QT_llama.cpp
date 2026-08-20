use crate::{TryReserveError, Vector};
use core::borrow::Borrow;
use core::fmt;

#[derive(Clone, Eq, PartialEq)]
pub struct OrderedMap<K, V> {
    entries: Vector<(K, V)>,
}

impl<K, V> Default for OrderedMap<K, V> {
    fn default() -> Self {
        Self {
            entries: Vector::new(),
        }
    }
}

impl<K: Ord, V> OrderedMap<K, V> {
    pub const fn new() -> Self {
        Self {
            entries: Vector::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Result<Self, TryReserveError> {
        Ok(Self {
            entries: Vector::with_capacity(capacity)?,
        })
    }

    /// Build a map from entries that are already sorted by key and unique.
    /// This avoids quadratic insertion shifts when loading large static maps.
    pub fn from_sorted_entries(entries: Vector<(K, V)>) -> Self {
        debug_assert!(entries.windows(2).all(|pair| pair[0].0 < pair[1].0));
        Self { entries }
    }

    pub fn try_insert(&mut self, key: K, value: V) -> Result<Option<V>, TryReserveError> {
        match self
            .entries
            .binary_search_by(|(existing, _)| existing.cmp(&key))
        {
            Ok(index) => Ok(Some(core::mem::replace(&mut self.entries[index].1, value))),
            Err(index) => {
                self.entries.try_insert(index, (key, value))?;
                Ok(None)
            }
        }
    }

    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        self.try_insert(key, value).expect("MRML allocation failed")
    }
}

impl<K, V> OrderedMap<K, V> {
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = (&K, &V)> {
        self.entries.iter().map(|(key, value)| (key, value))
    }
    pub fn iter_mut(&mut self) -> impl DoubleEndedIterator<Item = (&K, &mut V)> {
        self.entries.iter_mut().map(|(key, value)| (&*key, value))
    }
    pub fn get<Q: Ord + ?Sized>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
    {
        self.entries
            .binary_search_by(|(existing, _)| existing.borrow().cmp(key))
            .ok()
            .map(|index| &self.entries[index].1)
    }
    pub fn get_mut<Q: Ord + ?Sized>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
    {
        let index = self
            .entries
            .binary_search_by(|(existing, _)| existing.borrow().cmp(key))
            .ok()?;
        Some(&mut self.entries[index].1)
    }
}

impl<K: fmt::Debug, V: fmt::Debug> fmt::Debug for OrderedMap<K, V> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_map().entries(self.iter()).finish()
    }
}

impl<'a, K, V> IntoIterator for &'a OrderedMap<K, V> {
    type Item = (&'a K, &'a V);
    type IntoIter = core::iter::Map<core::slice::Iter<'a, (K, V)>, fn(&(K, V)) -> (&K, &V)>;
    fn into_iter(self) -> Self::IntoIter {
        fn pair<K, V>((key, value): &(K, V)) -> (&K, &V) {
            (key, value)
        }
        self.entries.iter().map(pair::<K, V>)
    }
}

impl<K, V> IntoIterator for OrderedMap<K, V> {
    type Item = (K, V);
    type IntoIter = crate::vector::IntoIter<(K, V)>;
    fn into_iter(self) -> Self::IntoIter {
        self.entries.into_iter()
    }
}

impl<K: Ord, V> core::iter::FromIterator<(K, V)> for OrderedMap<K, V> {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(entries: I) -> Self {
        let iterator = entries.into_iter();
        let mut output =
            Self::with_capacity(iterator.size_hint().0).expect("MRML allocation failed");
        for (key, value) in iterator {
            output.insert(key, value);
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inserts_in_key_order_and_replaces_values() {
        let mut map = OrderedMap::new();
        map.insert(3, "three");
        map.insert(1, "one");
        map.insert(2, "two");
        assert_eq!(map.insert(2, "second"), Some("two"));
        assert_eq!(map.get(&2), Some(&"second"));
        assert_eq!(
            map.iter()
                .map(|(key, _)| *key)
                .collect::<crate::Vector<_>>(),
            &[1, 2, 3][..]
        );
    }
}
