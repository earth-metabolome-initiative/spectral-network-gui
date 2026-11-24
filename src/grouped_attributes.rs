use std::collections::{BTreeSet, HashMap, HashSet};
use std::io::Read;

#[derive(Clone, Debug)]
pub struct GroupedAttributeMeans {
    pub columns: Vec<String>,
    pub axes: Vec<GroupAxis>,
    pub by_feature: HashMap<String, Vec<Option<f64>>>,
}

#[derive(Clone, Debug)]
pub struct GroupAxis {
    pub name: String,
    pub categories: Vec<String>,
    pub column_indices: Vec<usize>,
}

/// Aggregate peak areas for each feature and each sample metadata category.
/// Missing measurements are excluded from the mean; measured zeroes are included.
pub fn aggregate_grouped_attributes(
    quant: impl Read,
    metadata: impl Read,
) -> Result<GroupedAttributeMeans, String> {
    let mut metadata_csv = csv::ReaderBuilder::new().from_reader(metadata);
    let metadata_headers = metadata_csv
        .headers()
        .map_err(|err| format!("failed to read metadata header: {err}"))?
        .clone();
    let filename_col = unique_header(&metadata_headers, "filename", "metadata")?;
    let sample_type_col = optional_unique_header(&metadata_headers, "sample_type", "metadata")?;
    let mut attributes = Vec::new();
    let mut seen_names = HashSet::new();
    for (index, header) in metadata_headers.iter().enumerate() {
        if header.starts_with("ATTRIBUTE_") {
            if !seen_names.insert(header) {
                return Err(format!("duplicate metadata attribute header '{header}'"));
            }
            attributes.push((index, header.to_string()));
        }
    }

    // A sample may belong to one category on each axis. Keep category spelling
    // (after trimming) as the display label, and order categories deterministically.
    let mut categories: Vec<BTreeSet<String>> = vec![BTreeSet::new(); attributes.len()];
    let mut samples: HashMap<String, Vec<Option<String>>> = HashMap::new();
    for (row, record) in metadata_csv.records().enumerate() {
        let record = record.map_err(|err| format!("metadata row {}: {err}", row + 2))?;
        if sample_type_col.is_some_and(|col| {
            !record
                .get(col)
                .is_some_and(|value| value.trim().eq_ignore_ascii_case("sample"))
        }) {
            continue;
        }
        let filename = record.get(filename_col).unwrap_or_default().trim();
        if filename.is_empty() {
            return Err(format!("metadata row {} has an empty filename", row + 2));
        }
        let labels: Vec<Option<String>> = attributes
            .iter()
            .enumerate()
            .map(|(axis, (col, _))| {
                let label = record.get(*col).unwrap_or_default().trim();
                if label.is_empty()
                    || label.eq_ignore_ascii_case("NA")
                    || label.eq_ignore_ascii_case("N/A")
                {
                    None
                } else {
                    categories[axis].insert(label.to_string());
                    Some(label.to_string())
                }
            })
            .collect();
        if samples.insert(filename.to_string(), labels).is_some() {
            return Err(format!(
                "metadata row {} duplicates sample filename '{filename}'",
                row + 2
            ));
        }
    }

    let mut columns = Vec::new();
    let mut axes = Vec::new();
    let mut category_indices = Vec::new();
    let mut used_columns = HashSet::new();
    for ((_, name), labels) in attributes.iter().zip(categories) {
        let mut indices = HashMap::new();
        let mut axis = GroupAxis {
            name: name.clone(),
            categories: Vec::new(),
            column_indices: Vec::new(),
        };
        for label in labels {
            let column = format!("{name}:{label}");
            if !used_columns.insert(column.clone()) {
                return Err(format!("duplicate grouped attribute column '{column}'"));
            }
            let index = columns.len();
            columns.push(column);
            indices.insert(label.clone(), index);
            axis.categories.push(label);
            axis.column_indices.push(index);
        }
        category_indices.push(indices);
        axes.push(axis);
    }

    let mut quant_csv = csv::ReaderBuilder::new().from_reader(quant);
    let quant_headers = quant_csv
        .headers()
        .map_err(|err| format!("failed to read quantification header: {err}"))?
        .clone();
    let feature_col = unique_header(&quant_headers, "row ID", "quantification")?;
    let mut sample_columns = Vec::new();
    let mut seen_sample_columns = HashSet::new();
    for (col, header) in quant_headers.iter().enumerate() {
        if let Some(filename) = header.strip_suffix(" Peak area") {
            if let Some(labels) = samples.get(filename) {
                if !seen_sample_columns.insert(filename) {
                    return Err(format!(
                        "duplicate quantification peak area column '{header}'"
                    ));
                }
                let group_indices = labels
                    .iter()
                    .enumerate()
                    .filter_map(|(axis, label)| {
                        label
                            .as_ref()
                            .and_then(|label| category_indices[axis].get(label).copied())
                    })
                    .collect::<Vec<_>>();
                sample_columns.push((col, header.to_string(), group_indices));
            }
        }
    }

    let mut by_feature = HashMap::new();
    for (row, record) in quant_csv.records().enumerate() {
        let record = record.map_err(|err| format!("quantification row {}: {err}", row + 2))?;
        let feature = record.get(feature_col).unwrap_or_default().trim();
        if feature.is_empty() {
            return Err(format!(
                "quantification row {} has an empty row ID",
                row + 2
            ));
        }
        if by_feature.contains_key(feature) {
            return Err(format!(
                "quantification row {} duplicates row ID '{feature}'",
                row + 2
            ));
        }
        let mut sums = vec![0.0_f64; columns.len()];
        let mut counts = vec![0_usize; columns.len()];
        for (col, header, group_indices) in &sample_columns {
            let text = record.get(*col).unwrap_or_default().trim();
            if text.is_empty() {
                continue;
            }
            let value: f64 = text.parse().map_err(|_| {
                format!(
                    "quantification row {} ('{feature}'), column '{header}': invalid peak area '{text}'",
                    row + 2
                )
            })?;
            if !value.is_finite() || value < 0.0 {
                return Err(format!(
                    "quantification row {} ('{feature}'), column '{header}': peak area must be finite and nonnegative, got '{text}'",
                    row + 2
                ));
            }
            for &index in group_indices {
                sums[index] += value;
                if !sums[index].is_finite() {
                    return Err(format!(
                        "quantification row {} ('{feature}'), column '{}': peak area sum overflowed",
                        row + 2,
                        columns[index]
                    ));
                }
                counts[index] += 1;
            }
        }
        let means = sums
            .into_iter()
            .zip(counts)
            .map(|(sum, count)| {
                if count == 0 {
                    None
                } else {
                    Some(sum / count as f64)
                }
            })
            .collect();
        by_feature.insert(feature.to_string(), means);
    }
    Ok(GroupedAttributeMeans {
        columns,
        axes,
        by_feature,
    })
}

fn optional_unique_header(
    headers: &csv::StringRecord,
    name: &str,
    source: &str,
) -> Result<Option<usize>, String> {
    let mut matches = headers
        .iter()
        .enumerate()
        .filter(|(_, header)| *header == name);
    let first = matches.next().map(|(index, _)| index);
    if matches.next().is_some() {
        return Err(format!("duplicate {source} header '{name}'"));
    }
    Ok(first)
}

fn unique_header(headers: &csv::StringRecord, name: &str, source: &str) -> Result<usize, String> {
    optional_unique_header(headers, name, source)?
        .ok_or_else(|| format!("missing {source} header '{name}'"))
}

#[cfg(test)]
mod tests {
    use super::aggregate_grouped_attributes;

    #[test]
    fn aggregates_multiple_axes_with_sorted_categories_and_observed_zero() {
        let metadata = "filename,sample_type,ATTRIBUTE_site,ATTRIBUTE_treatment\na,sample, East ,treated\nb, sample ,East,control\nc,SAMPLE,West,treated\nd,QC,West,control\ne,sample,NA,N/A\nf,sample,West,control\n";
        let quant = "row ID,b Peak area,a Peak area,c Peak area,e Peak area,f Peak area,d Peak area,unmatched Peak area\nf1,4,0,8,, ,100,not-a-number\nf2,,0,,3,,100,not-a-number\nf3,,,,,,,\n";
        let result = aggregate_grouped_attributes(quant.as_bytes(), metadata.as_bytes()).unwrap();
        assert_eq!(
            result.columns,
            [
                "ATTRIBUTE_site:East",
                "ATTRIBUTE_site:West",
                "ATTRIBUTE_treatment:control",
                "ATTRIBUTE_treatment:treated"
            ]
        );
        assert_eq!(result.axes.len(), 2);
        assert_eq!(result.axes[0].name, "ATTRIBUTE_site");
        assert_eq!(result.axes[0].categories, ["East", "West"]);
        assert_eq!(result.axes[0].column_indices, [0, 1]);
        assert_eq!(result.axes[1].categories, ["control", "treated"]);
        assert_eq!(result.axes[1].column_indices, [2, 3]);
        assert_eq!(
            result.by_feature["f1"],
            [Some(2.0), Some(8.0), Some(4.0), Some(4.0)]
        );
        assert_eq!(result.by_feature["f2"], [Some(0.0), None, None, Some(0.0)]);
        assert_eq!(result.by_feature["f3"], [None, None, None, None]);
    }

    #[test]
    fn same_category_on_different_axes_has_distinct_columns() {
        let metadata = "filename,ATTRIBUTE_a,ATTRIBUTE_b\na,yes,yes\n";
        let quant = "row ID,a Peak area\nf,7\n";
        let result = aggregate_grouped_attributes(quant.as_bytes(), metadata.as_bytes()).unwrap();
        assert_eq!(result.columns, ["ATTRIBUTE_a:yes", "ATTRIBUTE_b:yes"]);
        assert_eq!(result.by_feature["f"], [Some(7.0), Some(7.0)]);
    }

    #[test]
    fn unmatched_sample_columns_leave_category_missing() {
        let metadata = "filename,ATTRIBUTE_site\na,East\nb,West\n";
        let quant = "row ID,a Peak area,other Peak area\nf,4,invalid\n";
        let result = aggregate_grouped_attributes(quant.as_bytes(), metadata.as_bytes()).unwrap();
        assert_eq!(result.by_feature["f"], [Some(4.0), None]);
    }

    #[test]
    fn rejects_duplicate_identities_and_invalid_observations() {
        let cases = [
            (
                "filename,ATTRIBUTE_x\na,X\na,X\n",
                "row ID,a Peak area\nf,1\n",
            ),
            (
                "filename,ATTRIBUTE_x\na,X\n",
                "row ID,a Peak area\nf,1\nf,2\n",
            ),
            (
                "filename,ATTRIBUTE_x\na,X\n",
                "row ID,a Peak area,a Peak area\nf,1,2\n",
            ),
            (
                "filename,ATTRIBUTE_x\na,X\n",
                "row ID,a Peak area\nf,nope\n",
            ),
            ("filename,ATTRIBUTE_x\na,X\n", "row ID,a Peak area\nf,-1\n"),
            ("filename,ATTRIBUTE_x\na,X\n", "row ID,a Peak area\nf,NaN\n"),
            ("filename,ATTRIBUTE_x\na,X\n", "row ID,a Peak area\nf,inf\n"),
        ];
        for (metadata, quant) in cases {
            assert!(
                aggregate_grouped_attributes(quant.as_bytes(), metadata.as_bytes()).is_err(),
                "invalid sample/feature inputs must not produce node intensities"
            );
        }
    }

    #[test]
    fn rejects_colliding_generated_columns() {
        let metadata = "filename,ATTRIBUTE_a,ATTRIBUTE_a:b\nx,b:c,c\n";
        let quant = "row ID,x Peak area\nf,1\n";
        assert!(aggregate_grouped_attributes(quant.as_bytes(), metadata.as_bytes()).is_err());
    }
}
