package com.briefbench.reference;

import java.util.*;

/**
 * Independent Okapi BM25 reference (the formula used by Lucene and rank-bm25),
 * reimplemented in Java to differentially test the Python BM25 arm
 * (membench.retrieval.bm25). We compare RANKING ORDER rather than absolute scores, so
 * the check is robust to the minor constant/IDF-floor differences between BM25
 * variants while still catching a real ordering bug in either implementation.
 */
public final class Bm25 {
    private final double k1;
    private final double b;

    public Bm25(double k1, double b) {
        this.k1 = k1;
        this.b = b;
    }

    public Bm25() {
        this(1.5, 0.75);
    }

    private static List<String> tokenize(String text) {
        List<String> out = new ArrayList<>();
        for (String t : text.toLowerCase(Locale.ROOT).split("[^a-z0-9]+")) {
            if (!t.isEmpty()) out.add(t);
        }
        return out;
    }

    /** Rank corpus ids by BM25 against the query, best first. */
    public List<String> rank(Map<String, String> corpus, String query) {
        List<String> ids = new ArrayList<>(corpus.keySet());
        Map<String, List<String>> docs = new HashMap<>();
        Map<String, Integer> df = new HashMap<>();
        double totalLen = 0.0;
        for (String id : ids) {
            List<String> toks = tokenize(corpus.get(id));
            docs.put(id, toks);
            totalLen += toks.size();
            for (String term : new HashSet<>(toks)) df.merge(term, 1, Integer::sum);
        }
        int n = ids.size();
        double avgdl = n == 0 ? 0.0 : totalLen / n;
        List<String> qTerms = tokenize(query);

        Map<String, Double> scores = new HashMap<>();
        for (String id : ids) {
            List<String> toks = docs.get(id);
            Map<String, Long> tf = new HashMap<>();
            for (String t : toks) tf.merge(t, 1L, Long::sum);
            double score = 0.0;
            for (String term : qTerms) {
                long f = tf.getOrDefault(term, 0L);
                if (f == 0) continue;
                double idf = Math.log(1 + (n - df.getOrDefault(term, 0) + 0.5) / (df.getOrDefault(term, 0) + 0.5));
                double denom = f + k1 * (1 - b + b * (avgdl == 0 ? 0 : toks.size() / avgdl));
                score += idf * (f * (k1 + 1)) / denom;
            }
            scores.put(id, score);
        }
        ids.sort((a, c) -> Double.compare(scores.get(c), scores.get(a)));
        return ids;
    }
}
