package dataloader;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * EvoSuite-derived tests adapted to JUnit 5 for TestDataLoader.
 */
class TestDataLoaderEvoSuiteTests {

    @Test
    void testCollectNoArgumentsCreatesEmptyEntry() {
        TestDataLoader loader = new TestDataLoader();

        loader.collectTestData();

        assertEquals(1, loader.getCollectedDataSize());
        assertEquals("", loader.getCollectedData(0));
    }

    @Test
    void testCollectNullValueSerialization() {
        TestDataLoader loader = new TestDataLoader();

        loader.collectTestData((Object) null);

        assertEquals(1, loader.getCollectedDataSize());
        assertEquals("null", loader.getCollectedData(0));
    }

    @Test
    void testCollectMixedPrimitiveAndStringValues() {
        TestDataLoader loader = new TestDataLoader();

        loader.collectTestData(1, "two", 3.0, true);

        assertEquals(1, loader.getCollectedDataSize());
        assertEquals("1,two,3.0,true", loader.getCollectedData(0));
    }

    @Test
    void testGetCollectedDataOnEmptyAndOutOfBounds() {
        TestDataLoader loader = new TestDataLoader();

        assertNull(loader.getCollectedData(0));
        loader.collectTestData("a", "b");
        assertNull(loader.getCollectedData(-1));
        assertNull(loader.getCollectedData(1));
    }

    @Test
    void testGetAllCollectedDataDefensiveCopy() {
        TestDataLoader loader = new TestDataLoader();
        loader.collectTestData("x", "y");

        List<String> snapshot = loader.getAllCollectedData();
        snapshot.clear();

        assertEquals(1, loader.getCollectedDataSize());
        assertEquals("x,y", loader.getCollectedData(0));
    }

    @Test
    void testExportToCsvRoundTrip() throws IOException {
        TestDataLoader loader = new TestDataLoader();
        loader.collectTestData(10, 20, 30);
        loader.collectTestData("alpha", "beta");

        Path output = Files.createTempFile("testdataloader-evosuite-", ".csv");
        try {
            loader.exportToCSV(output.toString());

            List<String> lines = Files.readAllLines(output);
            assertEquals(2, lines.size());
            assertEquals("10,20,30", lines.get(0));
            assertEquals("alpha,beta", lines.get(1));
        } finally {
            Files.deleteIfExists(output);
        }
    }

    @Test
    void testExportEmptyCollectionToCsvFromEvoSuite() throws IOException {
        TestDataLoader loader = new TestDataLoader();

        Path output = Files.createTempFile("testdataloader-empty-", ".csv");
        try {
            loader.exportToCSV(output.toString());
            List<String> lines = Files.readAllLines(output);
            assertTrue(lines.isEmpty());
        } finally {
            Files.deleteIfExists(output);
        }
    }

    @Test
    void testMutationAlignmentMathOperationsFromDataLoaderSuite() {
        assertEquals(6, MathOperations.sum(new int[]{1, 2, 3}));
        assertEquals(9, MathOperations.max(new int[]{1, 9, 3, 2}));
        assertEquals(-10, MathOperations.min(new int[]{-10, -5, -1, -8}));

        assertTrue(MathOperations.isPrime(29));
        assertFalse(MathOperations.isPrime(28));

        assertEquals(120, MathOperations.factorial(5));
        assertEquals(13, MathOperations.fibonacci(7));
        assertEquals(1, MathOperations.gcd(17, 13));

        assertTrue(MathOperations.isPerfectSquare(64));
        assertFalse(MathOperations.isPerfectSquare(63));

        assertTrue(MathOperations.isPalindrome("race car"));
        assertFalse(MathOperations.isPalindrome("software"));

        assertArrayEquals(new int[]{3, 2, 1}, MathOperations.reverseArray(new int[]{1, 2, 3}));
        assertArrayEquals(new int[]{1, 2, 3}, MathOperations.bubbleSort(new int[]{3, 1, 2}));
    }
}
