use needletail::{Sequence, parse_fastx_file};
use std::path::PathBuf;

// Read merged sequences from a genome file into single u8 vector
pub fn read_merge_seq(file_name: &PathBuf) -> Vec<u8> {
    let mut fna_seqs = Vec::<u8>::new();

    let mut fastx_reader = parse_fastx_file(file_name).expect("Opening .fna files failed");
    while let Some(record) = fastx_reader.next() {
        let seqrec = record.expect("invalid record");
        let norm_seq = seqrec.normalize(false);

        fna_seqs.push(b'N');
        fna_seqs.extend_from_slice(norm_seq.as_ref());
    }

    fna_seqs
}
