package com.briefbench.reference;

import org.junit.jupiter.api.Test;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

/** Cross-language agreement: same corpus/query as the Python BM25 suite; the two
 *  audit/export documents must outrank the UI documents (matching test_bm25.py). */
class Bm25Test {
    private static Map<String, String> corpus() {
        Map<String, String> c = new LinkedHashMap<>();
        c.put("D-1", "all data exports must call withAuditLog for SOC-2 compliance");
        c.put("D-2", "use DateRangePicker for date range selection in the UI");
        c.put("D-3", "every export endpoint requires an audit log entry before streaming");
        c.put("D-4", "primary actions use the Button component variant");
        return c;
    }

    @Test
    void ranksAuditDocsFirst() {
        List<String> ranked = new Bm25().rank(corpus(), "audit log on data export");
        assertTrue(Set.of("D-1", "D-3").contains(ranked.get(0)));
        assertEquals(Set.of("D-1", "D-3"), new HashSet<>(ranked.subList(0, 2)));
    }
}
