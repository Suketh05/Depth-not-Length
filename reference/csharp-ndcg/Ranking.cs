// Independent C# reference implementation of nDCG, used to differentially test the
// Python ranking scorer (membench.metrics.retrieval.ndcg_at_k). Binary relevance: an
// id is gold or it is not. If the two implementations disagree on the same inputs,
// the test fails -- catching a bug in either, which is the point of a cross-language
// reference.
namespace Membench;

public static class Ranking
{
    public static double NdcgAtK(IReadOnlyList<string> retrieved, ISet<string> gold, int? k = null)
    {
        if (gold.Count == 0) return 1.0;
        var cut = k.HasValue ? retrieved.Take(k.Value).ToList() : retrieved.ToList();
        double dcg = 0.0;
        for (int i = 0; i < cut.Count; i++)
            if (gold.Contains(cut[i]))
                dcg += 1.0 / Math.Log2(i + 2);
        int idealHits = k.HasValue ? Math.Min(gold.Count, cut.Count) : gold.Count;
        double idcg = 0.0;
        for (int r = 1; r <= idealHits; r++)
            idcg += 1.0 / Math.Log2(r + 1);
        return idcg > 0.0 ? dcg / idcg : 0.0;
    }
}
