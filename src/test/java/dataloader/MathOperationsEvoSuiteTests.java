package dataloader;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * EvoSuite-derived tests adapted to JUnit 5 so they run in the current pipeline.
 */
class MathOperationsEvoSuiteTests {

    @Test
    void testSumNullArrayFromEvoSuite() {
        assertEquals(0, MathOperations.sum(null));
    }

    @Test
    void testReverseArrayLengthPreserved() {
        int[] input = new int[8];
        int[] reversed = MathOperations.reverseArray(input);

        assertNotNull(reversed);
        assertEquals(8, reversed.length);
        assertNotSame(input, reversed);
    }

    @Test
    void testMinWithSparsePositiveArray() {
        int[] input = new int[4];
        input[0] = 976;
        assertEquals(0, MathOperations.min(input));
    }

    @Test
    void testMinThrowsOnNullArrayFromEvoSuite() {
        assertThrows(IllegalArgumentException.class, () -> MathOperations.min(null));
    }

    @Test
    void testMaxWithSparsePositiveArray() {
        int[] input = new int[7];
        input[1] = 266;
        assertEquals(266, MathOperations.max(input));
    }

    @Test
    void testMaxThrowsOnNullArrayFromEvoSuite() {
        assertThrows(IllegalArgumentException.class, () -> MathOperations.max(null));
    }

    @Test
    void testPrimeAndSquareSamplesFromEvoSuite() {
        assertFalse(MathOperations.isPrime(2291));
        assertFalse(MathOperations.isPrime(976));
        assertTrue(MathOperations.isPrime(3));

        assertFalse(MathOperations.isPerfectSquare(249));
        assertFalse(MathOperations.isPerfectSquare(-1666));
    }

    @Test
    void testPalindromeSamplesFromEvoSuite() {
        assertTrue(MathOperations.isPalindrome(""));
        assertFalse(MathOperations.isPalindrome(" Lw7NVW.L"));
    }

    @Test
    void testGcdAndEdgeFactorialsFromEvoSuite() {
        assertEquals(1, MathOperations.gcd(1, 1894));
        assertEquals(1, MathOperations.factorial(1));
        assertEquals(6, MathOperations.factorial(3));
    }

    @Test
    void testFibonacciSamplesFromEvoSuite() {
        assertEquals(1, MathOperations.fibonacci(1));
        assertEquals(1210320976, MathOperations.fibonacci(1884));
    }

    @Test
    void testBubbleSortSamplesFromEvoSuite() {
        int[] sorted = MathOperations.bubbleSort(new int[] {1884, 0, 0, 0, 0});
        assertArrayEquals(new int[] {0, 0, 0, 0, 1884}, sorted);
        assertNull(MathOperations.bubbleSort(null));
    }

    @Test
    void testMutationAlignmentAdditionalBoundariesFromEvoSuite() {
        assertEquals(0, MathOperations.sum(new int[0]));
        assertEquals(0, MathOperations.sum(new int[]{-5, 3, 2, -3, 3}));

        assertEquals(100, MathOperations.max(new int[]{-50, 0, 100, 25}));
        assertEquals(-50, MathOperations.min(new int[]{-50, 0, 100, 25}));

        assertTrue(MathOperations.isPrime(97));
        assertFalse(MathOperations.isPrime(1));
        assertFalse(MathOperations.isPrime(100));

        assertEquals(1, MathOperations.factorial(0));
        assertThrows(IllegalArgumentException.class, () -> MathOperations.factorial(-1));

        assertEquals(0, MathOperations.fibonacci(0));
        assertEquals(13, MathOperations.fibonacci(7));
        assertThrows(IllegalArgumentException.class, () -> MathOperations.fibonacci(-1));

        assertEquals(5, MathOperations.gcd(15, 10));

        assertTrue(MathOperations.isPerfectSquare(16));
        assertFalse(MathOperations.isPerfectSquare(15));

        assertTrue(MathOperations.isPalindrome("A man, a plan, a canal: Panama"));
        assertFalse(MathOperations.isPalindrome("not a palindrome"));

        int[] original = new int[] {5, 2, 8, 1, 9};
        int[] sorted = MathOperations.bubbleSort(original);
        assertArrayEquals(new int[] {1, 2, 5, 8, 9}, sorted);
        assertNotSame(original, sorted);
        assertArrayEquals(new int[] {5, 2, 8, 1, 9}, original);

        assertArrayEquals(new int[] {5, 4, 3, 2, 1}, MathOperations.reverseArray(new int[] {1, 2, 3, 4, 5}));
    }
}
