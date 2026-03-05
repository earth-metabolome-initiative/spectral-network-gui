use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct AttributeTable {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<String>>,
}

impl AttributeTable {
    pub fn parse_tsv(input: &str) -> Result<Self, String> {
        let mut lines = input.lines();
        let Some(header_line) = lines.find(|line| !line.trim().is_empty()) else {
            return Err("TSV file is empty".to_string());
        };

        let columns = split_tsv_line(header_line);
        if columns.is_empty() {
            return Err("TSV header has no columns".to_string());
        }

        let mut rows = Vec::new();
        for line in lines {
            if line.trim().is_empty() {
                continue;
            }
            let mut row = split_tsv_line(line);
            if row.len() < columns.len() {
                row.resize(columns.len(), String::new());
            } else if row.len() > columns.len() {
                row.truncate(columns.len());
            }
            rows.push(row);
        }

        Ok(Self { columns, rows })
    }
}

#[derive(Clone, Debug)]
pub struct LoadedAttributeTable {
    pub source_label: String,
    pub table: AttributeTable,
    key_column: usize,
    key_to_row: HashMap<String, usize>,
}

impl LoadedAttributeTable {
    pub fn new(source_label: String, table: AttributeTable) -> Self {
        let mut out = Self {
            source_label,
            table,
            key_column: 0,
            key_to_row: HashMap::new(),
        };
        out.rebuild_index();
        out
    }

    pub fn key_column(&self) -> usize {
        self.key_column
    }

    pub fn set_key_column(&mut self, key_column: usize) {
        if key_column < self.table.columns.len() {
            self.key_column = key_column;
            self.rebuild_index();
        }
    }

    pub fn find_row(&self, key: &str) -> Option<&Vec<String>> {
        let row_idx = self.key_to_row.get(key)?;
        self.table.rows.get(*row_idx)
    }

    pub fn find_row_index(&self, key: &str) -> Option<usize> {
        self.key_to_row.get(key).copied()
    }

    pub fn row(&self, row_idx: usize) -> Option<&Vec<String>> {
        self.table.rows.get(row_idx)
    }

    fn rebuild_index(&mut self) {
        self.key_to_row.clear();
        for (row_idx, row) in self.table.rows.iter().enumerate() {
            let Some(value) = row.get(self.key_column) else {
                continue;
            };
            let key = value.trim();
            if key.is_empty() || self.key_to_row.contains_key(key) {
                continue;
            }
            self.key_to_row.insert(key.to_string(), row_idx);
        }
    }
}

fn split_tsv_line(line: &str) -> Vec<String> {
    line.split('\t')
        .map(|s| s.trim_end_matches('\r').to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{AttributeTable, LoadedAttributeTable};

    #[test]
    fn parses_tsv_with_padding() {
        let input = "a\tb\tc\n1\t2\n3\t4\t5\t6\n";
        let table = AttributeTable::parse_tsv(input).expect("tsv should parse");
        assert_eq!(table.columns, vec!["a", "b", "c"]);
        assert_eq!(table.rows.len(), 2);
        assert_eq!(
            table.rows[0],
            vec!["1".to_string(), "2".to_string(), "".to_string()]
        );
        assert_eq!(
            table.rows[1],
            vec!["3".to_string(), "4".to_string(), "5".to_string()]
        );
    }

    #[test]
    fn key_column_selection_updates_index() {
        let input = "id\tname\n1\tfoo\n2\tbar\n";
        let table = AttributeTable::parse_tsv(input).expect("tsv should parse");
        let mut loaded = LoadedAttributeTable::new("test.tsv".to_string(), table);
        assert_eq!(
            loaded.find_row("1").expect("row for id"),
            &vec!["1".to_string(), "foo".to_string()]
        );
        loaded.set_key_column(1);
        assert_eq!(
            loaded.find_row("bar").expect("row for name"),
            &vec!["2".to_string(), "bar".to_string()]
        );
        assert!(loaded.find_row("2").is_none());
    }
}
