// Cross-language agreement: these expected values are identical to the Python
// suite's (tests/test_metrics_retrieval.py), so passing here AND there proves the two
// nDCG implementations agree.
using Xunit;

namespace Membench;

public class RankingTests
{
    [Fact]
    public void PerfectWhenGoldRankedFirst()
    {
        var r = Ranking.NdcgAtK(new[] { "g1", "g2", "x" }, new HashSet<string> { "g1", "g2" });
        Assert.Equal(1.0, r, 9);
    }

    [Fact]
    public void DiscountsLateHit()
    {
        var r = Ranking.NdcgAtK(new[] { "x", "g1" }, new HashSet<string> { "g1" });
        Assert.Equal(1.0 / Math.Log2(3), r, 9);
    }

    [Fact]
    public void NoGoldIsVacuouslyOne()
    {
        Assert.Equal(1.0, Ranking.NdcgAtK(new[] { "x" }, new HashSet<string>()));
    }
}
