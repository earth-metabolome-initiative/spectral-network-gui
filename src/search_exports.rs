use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use crate::attributes::AttributeTable;
use crate::export::SearchQueryKey;

#[derive(Clone, Debug, PartialEq)]
pub struct SearchExportRow {
    pub query_export_key: String,
    pub short_inchikey: Option<String>,
    pub hit_rank: usize,
    pub hit_spectral_score: f64,
    pub hit_combined_score: f64,
    values: BTreeMap<String, String>,
}

impl SearchExportRow {
    pub fn value(&self, column: &str) -> Option<&str> {
        self.values.get(column).map(String::as_str)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SearchExportTable {
    pub source_label: String,
    pub rows: Vec<SearchExportRow>,
    pub hit_columns: Vec<String>,
    pub inferred_query_key: Option<SearchQueryKey>,
}

#[derive(Clone, Copy, Debug)]
pub struct SearchExportJobInput<'a> {
    pub alias: &'a str,
    pub max_rank: usize,
    pub table: &'a SearchExportTable,
}

#[derive(Clone, Debug)]
pub struct MergedSearchExportTable {
    pub table: AttributeTable,
    pub inferred_query_key: Option<SearchQueryKey>,
    pub library_aliases: Vec<String>,
    pub library_prefixes: Vec<String>,
}

pub fn parse_search_export_tsv(
    source_label: &str,
    text: &str,
) -> Result<SearchExportTable, String> {
    let table = AttributeTable::parse_tsv(text)
        .map_err(|err| format!("failed to parse search export TSV {source_label}: {err}"))?;

    let query_export_key_idx = header_index(&table.columns, "query_export_key")?;
    let hit_rank_idx = header_index(&table.columns, "hit_rank")?;
    let hit_spectral_score_idx = header_index(&table.columns, "hit_spectral_score")?;
    let hit_combined_score_idx = header_index(&table.columns, "hit_combined_score")?;
    let short_inchikey_idx = header_index(&table.columns, "hit_taxonomic_short_inchikey")?;

    let hit_columns = table
        .columns
        .iter()
        .filter(|column| normalized_column_name(column).starts_with("hit"))
        .cloned()
        .collect::<Vec<_>>();
    let inferred_query_key = infer_query_key_mode(&table);

    let mut rows = Vec::with_capacity(table.rows.len());
    for (row_idx, row) in table.rows.iter().enumerate() {
        let values = table
            .columns
            .iter()
            .cloned()
            .zip(row.iter().cloned())
            .collect::<BTreeMap<_, _>>();
        let query_export_key = row
            .get(query_export_key_idx)
            .map(String::as_str)
            .map(str::trim)
            .unwrap_or_default()
            .to_string();
        let short_inchikey = row
            .get(short_inchikey_idx)
            .map(String::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string);
        let hit_rank = parse_usize_field(
            row.get(hit_rank_idx).map(String::as_str),
            "hit_rank",
            source_label,
            row_idx,
        )?;
        let hit_spectral_score = parse_f64_field(
            row.get(hit_spectral_score_idx).map(String::as_str),
            "hit_spectral_score",
            source_label,
            row_idx,
        )?;
        let hit_combined_score = parse_f64_field(
            row.get(hit_combined_score_idx).map(String::as_str),
            "hit_combined_score",
            source_label,
            row_idx,
        )?;

        rows.push(SearchExportRow {
            query_export_key,
            short_inchikey,
            hit_rank,
            hit_spectral_score,
            hit_combined_score,
            values,
        });
    }

    Ok(SearchExportTable {
        source_label: source_label.to_string(),
        rows,
        hit_columns,
        inferred_query_key,
    })
}

pub fn merge_search_exports(
    jobs: &[SearchExportJobInput<'_>],
) -> Result<MergedSearchExportTable, String> {
    if jobs.len() < 2 {
        return Err("load at least two search export TSVs before merging".to_string());
    }

    let mut known_query_modes = BTreeSet::new();
    for job in jobs {
        if let Some(mode) = job.table.inferred_query_key {
            known_query_modes.insert(mode.label().to_string());
        }
    }
    if known_query_modes.len() > 1 {
        return Err(format!(
            "loaded search export TSVs use incompatible inferred query key modes: {}",
            known_query_modes.into_iter().collect::<Vec<_>>().join(", ")
        ));
    }

    let mut alias_prefixes = HashSet::new();
    let mut job_prefixes = Vec::with_capacity(jobs.len());
    for job in jobs {
        let prefix = sanitize_alias(job.alias);
        if !alias_prefixes.insert(prefix.clone()) {
            return Err(format!(
                "duplicate merged search alias '{prefix}'; rename one of the loaded exports"
            ));
        }
        job_prefixes.push(prefix);
    }

    let per_job_best_rows = jobs
        .iter()
        .map(|job| best_rows_for_job(job.table, job.max_rank.max(1)))
        .collect::<Vec<_>>();

    let mut all_keys = BTreeSet::new();
    for rows in &per_job_best_rows {
        all_keys.extend(rows.keys().cloned());
    }

    let mut header = vec![
        "query_export_key".to_string(),
        "query_node_id".to_string(),
        "query_feature_id".to_string(),
        "query_featurelist_feature_id".to_string(),
        "query_scans".to_string(),
        "query_label".to_string(),
        "structure_short_inchikey".to_string(),
        "structure_name".to_string(),
        "structure_smiles".to_string(),
    ];

    let prefixed_hit_columns = jobs
        .iter()
        .zip(job_prefixes.iter())
        .map(|(job, prefix)| {
            let mut columns = vec![format!("{prefix}__present")];
            columns.extend(
                job.table
                    .hit_columns
                    .iter()
                    .filter(|column| normalized_column_name(column) != "hittaxonomicshortinchikey")
                    .map(|column| {
                        format!(
                            "{prefix}__{}",
                            column.strip_prefix("hit_").unwrap_or(column.as_str())
                        )
                    })
                    .collect::<Vec<_>>(),
            );
            columns
        })
        .collect::<Vec<_>>();
    for columns in &prefixed_hit_columns {
        header.extend(columns.iter().cloned());
    }

    let mut rows = Vec::new();
    for (query_export_key, short_inchikey) in all_keys {
        let matched_rows = per_job_best_rows
            .iter()
            .map(|rows_by_key| {
                rows_by_key
                    .get(&(query_export_key.clone(), short_inchikey.clone()))
                    .copied()
            })
            .collect::<Vec<_>>();

        let mut row = vec![
            query_export_key.clone(),
            pick_first_value(&matched_rows, "query_node_id"),
            pick_first_value(&matched_rows, "query_feature_id"),
            pick_first_value(&matched_rows, "query_featurelist_feature_id"),
            pick_first_value(&matched_rows, "query_scans"),
            pick_first_value(&matched_rows, "query_label"),
            short_inchikey.clone(),
            pick_structure_name(&matched_rows),
            pick_structure_smiles(&matched_rows),
        ];

        for (matched_row, hit_columns) in matched_rows.iter().zip(jobs.iter().map(|job| {
            job.table
                .hit_columns
                .iter()
                .filter(|column| normalized_column_name(column) != "hittaxonomicshortinchikey")
                .cloned()
                .collect::<Vec<_>>()
        })) {
            row.push(if matched_row.is_some() { "1" } else { "0" }.to_string());
            for hit_column in hit_columns {
                row.push(match matched_row {
                    Some(matched_row) => matched_row
                        .value(&hit_column)
                        .map(str::trim)
                        .unwrap_or_default()
                        .to_string(),
                    None => String::new(),
                });
            }
        }
        rows.push(row);
    }

    Ok(MergedSearchExportTable {
        table: AttributeTable {
            columns: header,
            rows,
        },
        inferred_query_key: jobs.first().and_then(|job| job.table.inferred_query_key),
        library_aliases: jobs.iter().map(|job| job.alias.to_string()).collect(),
        library_prefixes: job_prefixes,
    })
}

pub fn search_export_alias_prefix(alias: &str) -> String {
    sanitize_alias(alias)
}

pub fn default_search_export_alias(source_label: &str) -> String {
    let stem = std::path::Path::new(source_label)
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or(source_label);
    let alias = sanitize_alias(stem);
    if alias.is_empty() {
        "library".to_string()
    } else {
        alias
    }
}

fn header_index(columns: &[String], target: &str) -> Result<usize, String> {
    columns
        .iter()
        .position(|column| normalized_column_name(column) == normalized_column_name(target))
        .ok_or_else(|| format!("TSV is missing required column '{target}'"))
}

fn parse_usize_field(
    value: Option<&str>,
    field: &str,
    source_label: &str,
    row_idx: usize,
) -> Result<usize, String> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("missing {field} in {source_label} row {}", row_idx + 2))?
        .parse::<usize>()
        .map_err(|err| {
            format!(
                "invalid {field} in {source_label} row {}: {err}",
                row_idx + 2
            )
        })
}

fn parse_f64_field(
    value: Option<&str>,
    field: &str,
    source_label: &str,
    row_idx: usize,
) -> Result<f64, String> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("missing {field} in {source_label} row {}", row_idx + 2))?
        .parse::<f64>()
        .map_err(|err| {
            format!(
                "invalid {field} in {source_label} row {}: {err}",
                row_idx + 2
            )
        })
}

fn infer_query_key_mode(table: &AttributeTable) -> Option<SearchQueryKey> {
    let export_idx = table
        .columns
        .iter()
        .position(|column| normalized_column_name(column) == "queryexportkey")?;

    for mode in SearchQueryKey::ALL {
        let candidate_column = match mode {
            SearchQueryKey::FeatureId => "query_feature_id",
            SearchQueryKey::FeaturelistFeatureId => "query_featurelist_feature_id",
            SearchQueryKey::Scans => "query_scans",
            SearchQueryKey::RawName => "query_raw_name",
            SearchQueryKey::Label => "query_label",
            SearchQueryKey::NodeId => "query_node_id",
        };
        let Some(candidate_idx) = table.columns.iter().position(|column| {
            normalized_column_name(column) == normalized_column_name(candidate_column)
        }) else {
            continue;
        };

        let mut compared = 0usize;
        let mut all_match = true;
        for row in &table.rows {
            let exported = row
                .get(export_idx)
                .map(String::as_str)
                .map(str::trim)
                .unwrap_or_default();
            let candidate = row
                .get(candidate_idx)
                .map(String::as_str)
                .map(str::trim)
                .unwrap_or_default();
            if exported.is_empty() || candidate.is_empty() {
                continue;
            }
            compared += 1;
            if exported != candidate {
                all_match = false;
                break;
            }
        }
        if compared > 0 && all_match {
            return Some(mode);
        }
    }
    None
}

fn sanitize_alias(alias: &str) -> String {
    let mut out = String::new();
    let mut last_was_sep = false;
    for ch in alias.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            last_was_sep = false;
        } else if !last_was_sep {
            out.push('_');
            last_was_sep = true;
        }
    }
    out.trim_matches('_').to_string()
}

fn best_rows_for_job<'a>(
    table: &'a SearchExportTable,
    max_rank: usize,
) -> HashMap<(String, String), &'a SearchExportRow> {
    let mut best: HashMap<(String, String), &'a SearchExportRow> = HashMap::new();
    for row in &table.rows {
        let Some(short_inchikey) = row.short_inchikey.as_ref() else {
            continue;
        };
        if row.query_export_key.trim().is_empty() || row.hit_rank > max_rank {
            continue;
        }
        let key = (row.query_export_key.clone(), short_inchikey.clone());
        let replace = match best.get(&key) {
            None => true,
            Some(current) => is_better_row(row, current),
        };
        if replace {
            best.insert(key, row);
        }
    }
    best
}

fn is_better_row(candidate: &SearchExportRow, current: &SearchExportRow) -> bool {
    candidate.hit_rank < current.hit_rank
        || (candidate.hit_rank == current.hit_rank
            && candidate.hit_combined_score > current.hit_combined_score)
        || (candidate.hit_rank == current.hit_rank
            && candidate.hit_combined_score == current.hit_combined_score
            && candidate.hit_spectral_score > current.hit_spectral_score)
}

fn pick_first_value(rows: &[Option<&SearchExportRow>], column: &str) -> String {
    rows.iter()
        .find_map(|row| {
            let row = row.as_ref()?;
            row.value(column)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_string)
        })
        .unwrap_or_default()
}

fn pick_structure_name(rows: &[Option<&SearchExportRow>]) -> String {
    const CANDIDATES: [&str; 4] = [
        "hit_structure_nameTraditional",
        "hit_structure_name",
        "hit_compound_name",
        "hit_raw_name",
    ];
    for column in CANDIDATES {
        let value = pick_first_value(rows, column);
        if !value.is_empty() {
            return value;
        }
    }
    String::new()
}

fn pick_structure_smiles(rows: &[Option<&SearchExportRow>]) -> String {
    const CANDIDATES: [&str; 2] = ["hit_SMILES", "hit_smiles"];
    for column in CANDIDATES {
        let value = pick_first_value(rows, column);
        if !value.is_empty() {
            return value;
        }
    }
    String::new()
}

fn normalized_column_name(value: &str) -> String {
    value
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .map(|ch| ch.to_ascii_lowercase())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{
        SearchExportJobInput, default_search_export_alias, merge_search_exports,
        parse_search_export_tsv,
    };
    use crate::export::SearchQueryKey;

    fn column_idx(columns: &[String], name: &str) -> usize {
        columns
            .iter()
            .position(|column| column == name)
            .expect("column should exist")
    }

    fn sample_export(rows: &[&str]) -> String {
        let mut out = String::from(
            "query_export_key\tquery_node_id\tquery_feature_id\tquery_featurelist_feature_id\tquery_scans\tquery_label\tquery_raw_name\thit_rank\thit_spectral_score\thit_taxonomic_score\thit_combined_score\thit_matches\thit_taxonomic_shared_rank\thit_taxonomic_organism_name\thit_taxonomic_organism_wikidata\thit_taxonomic_short_inchikey\thit_precursor_mz\thit_raw_name\thit_SMILES\n",
        );
        for row in rows {
            out.push_str(row);
            out.push('\n');
        }
        out
    }

    #[test]
    fn parses_search_export_rows_and_infers_query_mode() {
        let text = sample_export(&[
            "feat1\t1\tfeat1\t\t\tlabel1\traw1\t1\t0.9\t0\t0.9\t5\t\t\t\tABCDEFGHIJKLMN\t100.0\thit1\tCCO",
        ]);
        let parsed = parse_search_export_tsv("a.tsv", &text).expect("search export");
        assert_eq!(parsed.rows.len(), 1);
        assert_eq!(parsed.inferred_query_key, Some(SearchQueryKey::FeatureId));
        assert_eq!(
            parsed.rows[0].short_inchikey.as_deref(),
            Some("ABCDEFGHIJKLMN")
        );
    }

    #[test]
    fn merge_keeps_same_query_and_same_structure_only() {
        let lotus = parse_search_export_tsv(
            "lotus.tsv",
            &sample_export(&[
                "feat1\t1\tfeat1\t\t\tlabel1\traw1\t1\t0.9\t0\t0.9\t5\t\t\t\tABCDEFGHIJKLMN\t100.0\tlotus_hit\tCCO",
            ]),
        )
        .expect("lotus");
        let gnps = parse_search_export_tsv(
            "gnps.tsv",
            &sample_export(&[
                "feat1\t1\tfeat1\t\t\tlabel1\traw1\t3\t0.7\t0\t0.7\t4\t\t\t\tABCDEFGHIJKLMN\t100.0\tgnps_hit\tCCO",
                "feat1\t1\tfeat1\t\t\tlabel1\traw1\t1\t0.8\t0\t0.8\t6\t\t\t\tZZZZZZZZZZZZZZ\t100.0\tother\tCCC",
            ]),
        )
        .expect("gnps");

        let merged = merge_search_exports(&[
            SearchExportJobInput {
                alias: "LOTUS",
                max_rank: 5,
                table: &lotus,
            },
            SearchExportJobInput {
                alias: "GNPS",
                max_rank: 5,
                table: &gnps,
            },
        ])
        .expect("merged");

        let short_inchikey_idx = column_idx(&merged.table.columns, "structure_short_inchikey");
        let lotus_present_idx = column_idx(&merged.table.columns, "lotus__present");
        let gnps_present_idx = column_idx(&merged.table.columns, "gnps__present");

        assert_eq!(merged.table.rows.len(), 2);
        assert_eq!(merged.table.rows[0][0], "feat1");
        assert_eq!(merged.table.rows[0][short_inchikey_idx], "ABCDEFGHIJKLMN");
        assert_eq!(merged.table.rows[0][lotus_present_idx], "1");
        assert_eq!(merged.table.rows[0][gnps_present_idx], "1");
        assert_eq!(merged.table.rows[1][short_inchikey_idx], "ZZZZZZZZZZZZZZ");
        assert_eq!(merged.table.rows[1][lotus_present_idx], "0");
        assert_eq!(merged.table.rows[1][gnps_present_idx], "1");
    }

    #[test]
    fn merge_prefers_best_duplicate_row_by_rank_and_scores() {
        let lotus = parse_search_export_tsv(
            "lotus.tsv",
            &sample_export(&[
                "feat1\t1\tfeat1\t\t\tlabel1\traw1\t2\t0.8\t0\t0.8\t5\t\t\t\tABCDEFGHIJKLMN\t100.0\tworse\tCCO",
                "feat1\t1\tfeat1\t\t\tlabel1\traw1\t1\t0.7\t0\t0.7\t5\t\t\t\tABCDEFGHIJKLMN\t100.0\tbetter_rank\tCCO",
                "feat1\t1\tfeat1\t\t\tlabel1\traw1\t1\t0.9\t0\t0.9\t5\t\t\t\tABCDEFGHIJKLMN\t100.0\tbest_score\tCCO",
            ]),
        )
        .expect("lotus");
        let gnps = parse_search_export_tsv(
            "gnps.tsv",
            &sample_export(&[
                "feat1\t1\tfeat1\t\t\tlabel1\traw1\t1\t0.7\t0\t0.7\t4\t\t\t\tABCDEFGHIJKLMN\t100.0\tgnps_hit\tCCO",
            ]),
        )
        .expect("gnps");

        let merged = merge_search_exports(&[
            SearchExportJobInput {
                alias: "LOTUS",
                max_rank: 5,
                table: &lotus,
            },
            SearchExportJobInput {
                alias: "GNPS",
                max_rank: 5,
                table: &gnps,
            },
        ])
        .expect("merged");

        assert!(
            merged
                .table
                .columns
                .iter()
                .any(|column| column == "lotus__raw_name")
        );
        let raw_name_idx = merged
            .table
            .columns
            .iter()
            .position(|column| column == "lotus__raw_name")
            .expect("raw name col");
        assert_eq!(merged.table.rows[0][raw_name_idx], "best_score");
    }

    #[test]
    fn merge_excludes_rows_missing_short_inchikey() {
        let lotus = parse_search_export_tsv(
            "lotus.tsv",
            &sample_export(&[
                "feat1\t1\tfeat1\t\t\tlabel1\traw1\t1\t0.9\t0\t0.9\t5\t\t\t\t\t100.0\thit1\tCCO",
            ]),
        )
        .expect("lotus");
        let gnps = parse_search_export_tsv(
            "gnps.tsv",
            &sample_export(&[
                "feat1\t1\tfeat1\t\t\tlabel1\traw1\t1\t0.7\t0\t0.7\t4\t\t\t\tABCDEFGHIJKLMN\t100.0\thit2\tCCO",
            ]),
        )
        .expect("gnps");

        let merged = merge_search_exports(&[
            SearchExportJobInput {
                alias: "LOTUS",
                max_rank: 5,
                table: &lotus,
            },
            SearchExportJobInput {
                alias: "GNPS",
                max_rank: 5,
                table: &gnps,
            },
        ])
        .expect("merged");

        let short_inchikey_idx = column_idx(&merged.table.columns, "structure_short_inchikey");
        let lotus_present_idx = column_idx(&merged.table.columns, "lotus__present");
        let gnps_present_idx = column_idx(&merged.table.columns, "gnps__present");

        assert_eq!(merged.table.rows.len(), 1);
        assert_eq!(merged.table.rows[0][short_inchikey_idx], "ABCDEFGHIJKLMN");
        assert_eq!(merged.table.rows[0][lotus_present_idx], "0");
        assert_eq!(merged.table.rows[0][gnps_present_idx], "1");
    }

    #[test]
    fn merge_applies_per_library_max_rank_filters() {
        let lotus = parse_search_export_tsv(
            "lotus.tsv",
            &sample_export(&[
                "feat1\t1\tfeat1\t\t\tlabel1\traw1\t1\t0.9\t0\t0.9\t5\t\t\t\tABCDEFGHIJKLMN\t100.0\tlotus_hit\tCCO",
            ]),
        )
        .expect("lotus");
        let gnps = parse_search_export_tsv(
            "gnps.tsv",
            &sample_export(&[
                "feat1\t1\tfeat1\t\t\tlabel1\traw1\t3\t0.7\t0\t0.7\t4\t\t\t\tABCDEFGHIJKLMN\t100.0\tgnps_hit\tCCO",
            ]),
        )
        .expect("gnps");

        let merged = merge_search_exports(&[
            SearchExportJobInput {
                alias: "LOTUS",
                max_rank: 1,
                table: &lotus,
            },
            SearchExportJobInput {
                alias: "GNPS",
                max_rank: 3,
                table: &gnps,
            },
        ])
        .expect("merged");
        assert_eq!(merged.table.rows.len(), 1);

        let filtered_out = merge_search_exports(&[
            SearchExportJobInput {
                alias: "LOTUS",
                max_rank: 1,
                table: &lotus,
            },
            SearchExportJobInput {
                alias: "GNPS",
                max_rank: 2,
                table: &gnps,
            },
        ])
        .expect("merged");
        let lotus_present_idx = column_idx(&filtered_out.table.columns, "lotus__present");
        let gnps_present_idx = column_idx(&filtered_out.table.columns, "gnps__present");

        assert_eq!(filtered_out.table.rows.len(), 1);
        assert_eq!(filtered_out.table.rows[0][lotus_present_idx], "1");
        assert_eq!(filtered_out.table.rows[0][gnps_present_idx], "0");
    }

    #[test]
    fn merged_table_drops_query_raw_name_column() {
        let lotus = parse_search_export_tsv(
            "lotus.tsv",
            &sample_export(&[
                "feat1\t1\tfeat1\t\t\tlabel1\traw1\t1\t0.9\t0\t0.9\t5\t\t\t\tABCDEFGHIJKLMN\t100.0\tlotus_hit\tCCO",
            ]),
        )
        .expect("lotus");
        let gnps = parse_search_export_tsv(
            "gnps.tsv",
            &sample_export(&[
                "feat1\t1\tfeat1\t\t\tlabel1\traw1\t1\t0.7\t0\t0.7\t4\t\t\t\tABCDEFGHIJKLMN\t100.0\tgnps_hit\tCCO",
            ]),
        )
        .expect("gnps");

        let merged = merge_search_exports(&[
            SearchExportJobInput {
                alias: "LOTUS",
                max_rank: 5,
                table: &lotus,
            },
            SearchExportJobInput {
                alias: "GNPS",
                max_rank: 5,
                table: &gnps,
            },
        ])
        .expect("merged");

        assert!(
            !merged
                .table
                .columns
                .iter()
                .any(|column| column == "query_raw_name")
        );
    }

    #[test]
    fn alias_defaults_to_sanitized_stem() {
        assert_eq!(
            default_search_export_alias("/tmp/My LOTUS export.tsv"),
            "my_lotus_export"
        );
    }
}
