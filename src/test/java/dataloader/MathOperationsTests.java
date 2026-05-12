package dataloader;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Automated Tests for MathOperations class
 * Generated with EvoSuite-inspired approach for comprehensive coverage
 */
class MathOperationsTests {
    
    // ============ Sum Tests ============
    @Test
    void testSumEmptyArray() {
        assertEquals(0, MathOperations.sum(new int[0]));
    }
    
    @Test
    void testSumNullArray() {
        assertEquals(0, MathOperations.sum(null));
    }
    
    @Test
    void testSumSingleElement() {
        assertEquals(5, MathOperations.sum(new int[]{5}));
    }
    
    @Test
    void testSumPositiveNumbers() {
        assertEquals(15, MathOperations.sum(new int[]{1, 2, 3, 4, 5}));
    }
    
    @Test
    void testSumNegativeNumbers() {
        assertEquals(-15, MathOperations.sum(new int[]{-1, -2, -3, -4, -5}));
    }
    
    @Test
    void testSumMixedNumbers() {
        assertEquals(0, MathOperations.sum(new int[]{-5, 3, 2, -3, 3}));
    }
    
    // ============ Max Tests ============
    @Test
    void testMaxSingleElement() {
        assertEquals(42, MathOperations.max(new int[]{42}));
    }
    
    @Test
    void testMaxPositiveNumbers() {
        assertEquals(9, MathOperations.max(new int[]{1, 5, 9, 3, 2}));
    }
    
    @Test
    void testMaxNegativeNumbers() {
        assertEquals(-1, MathOperations.max(new int[]{-10, -5, -1, -8}));
    }
    
    @Test
    void testMaxMixedNumbers() {
        assertEquals(100, MathOperations.max(new int[]{-50, 0, 100, 25}));
    }
    
    @Test
    void testMaxThrowsOnEmpty() {
        assertThrows(IllegalArgumentException.class, () -> MathOperations.max(new int[0]));
    }
    
    @Test
    void testMaxThrowsOnNull() {
        assertThrows(IllegalArgumentException.class, () -> MathOperations.max(null));
    }
    
    // ============ Min Tests ============
    @Test
    void testMinSingleElement() {
        assertEquals(42, MathOperations.min(new int[]{42}));
    }
    
    @Test
    void testMinPositiveNumbers() {
        assertEquals(1, MathOperations.min(new int[]{1, 5, 9, 3, 2}));
    }
    
    @Test
    void testMinNegativeNumbers() {
        assertEquals(-10, MathOperations.min(new int[]{-10, -5, -1, -8}));
    }
    
    @Test
    void testMinMixedNumbers() {
        assertEquals(-50, MathOperations.min(new int[]{-50, 0, 100, 25}));
    }
    
    // ============ isPrime Tests ============
    @Test
    void testIsPrimeSmallPrimes() {
        assertTrue(MathOperations.isPrime(2));
        assertTrue(MathOperations.isPrime(3));
        assertTrue(MathOperations.isPrime(5));
        assertTrue(MathOperations.isPrime(7));
    }
    
    @Test
    void testIsPrimeLargePrimes() {
        assertTrue(MathOperations.isPrime(97));
        assertTrue(MathOperations.isPrime(101));
    }
    
    @Test
    void testIsPrimeComposite() {
        assertFalse(MathOperations.isPrime(4));
        assertFalse(MathOperations.isPrime(9));
        assertFalse(MathOperations.isPrime(10));
        assertFalse(MathOperations.isPrime(25));
        assertFalse(MathOperations.isPrime(49));
    }
    
    @Test
    void testIsPrimeEdgeCases() {
        assertFalse(MathOperations.isPrime(0));
        assertFalse(MathOperations.isPrime(1));
        assertFalse(MathOperations.isPrime(-5));
    }
    
    // ============ Factorial Tests ============
    @Test
    void testFactorialZero() {
        assertEquals(1, MathOperations.factorial(0));
    }
    
    @Test
    void testFactorialOne() {
        assertEquals(1, MathOperations.factorial(1));
    }
    
    @Test
    void testFactorialSmallNumbers() {
        assertEquals(2, MathOperations.factorial(2));
        assertEquals(6, MathOperations.factorial(3));
        assertEquals(24, MathOperations.factorial(4));
        assertEquals(120, MathOperations.factorial(5));
    }
    
    @Test
    void testFactorialNegative() {
        assertThrows(IllegalArgumentException.class, () -> MathOperations.factorial(-1));
    }
    
    // ============ isPalindrome Tests ============
    @Test
    void testIsPalindromeSimple() {
        assertTrue(MathOperations.isPalindrome("aba"));
        assertTrue(MathOperations.isPalindrome("racecar"));
        assertFalse(MathOperations.isPalindrome("hello"));
    }
    
    @Test
    void testIsPalindromeWithSpaces() {
        assertTrue(MathOperations.isPalindrome("a b a"));
        assertTrue(MathOperations.isPalindrome("race car"));
    }

    @Test
    void testIsPalindromeIgnoresCaseAndPunctuation() {
        assertTrue(MathOperations.isPalindrome("A man, a plan, a canal: Panama!"));
        assertTrue(MathOperations.isPalindrome("No 'x' in Nixon"));
        assertFalse(MathOperations.isPalindrome("Panama canal"));
    }
    
    @Test
    void testIsPalindromeEdgeCases() {
        assertTrue(MathOperations.isPalindrome(""));
        assertTrue(MathOperations.isPalindrome("a"));
        assertTrue(MathOperations.isPalindrome(null));
    }
    
    // ============ Fibonacci Tests ============
    @Test
    void testFibonacciZero() {
        assertEquals(0, MathOperations.fibonacci(0));
    }
    
    @Test
    void testFibonacciOne() {
        assertEquals(1, MathOperations.fibonacci(1));
    }
    
    @Test
    void testFibonacciSequence() {
        assertEquals(1, MathOperations.fibonacci(2));
        assertEquals(2, MathOperations.fibonacci(3));
        assertEquals(3, MathOperations.fibonacci(4));
        assertEquals(5, MathOperations.fibonacci(5));
        assertEquals(8, MathOperations.fibonacci(6));
    }
    
    @Test
    void testFibonacciNegative() {
        assertThrows(IllegalArgumentException.class, () -> MathOperations.fibonacci(-1));
    }
    
    // ============ GCD Tests ============
    @Test
    void testGcdSimple() {
        assertEquals(3, MathOperations.gcd(9, 6));
        assertEquals(5, MathOperations.gcd(15, 10));
    }
    
    @Test
    void testGcdCoprime() {
        assertEquals(1, MathOperations.gcd(7, 5));
        assertEquals(1, MathOperations.gcd(17, 13));
    }
    
    @Test
    void testGcdWithZero() {
        assertEquals(5, MathOperations.gcd(5, 0));
        assertEquals(5, MathOperations.gcd(0, 5));
        assertEquals(6, MathOperations.gcd(54, 24));
        assertEquals(6, MathOperations.gcd(24, 54));
    }
    
    // ============ isPerfectSquare Tests ============
    @Test
    void testIsPerfectSquareTrue() {
        assertTrue(MathOperations.isPerfectSquare(0));
        assertTrue(MathOperations.isPerfectSquare(1));
        assertTrue(MathOperations.isPerfectSquare(4));
        assertTrue(MathOperations.isPerfectSquare(9));
        assertTrue(MathOperations.isPerfectSquare(16));
    }
    
    @Test
    void testIsPerfectSquareFalse() {
        assertFalse(MathOperations.isPerfectSquare(2));
        assertFalse(MathOperations.isPerfectSquare(3));
        assertFalse(MathOperations.isPerfectSquare(5));
    }
    
    @Test
    void testIsPerfectSquareNegative() {
        assertFalse(MathOperations.isPerfectSquare(-1));
        assertFalse(MathOperations.isPerfectSquare(-4));
    }
    
    // ============ reverseArray Tests ============
    @Test
    void testReverseArraySimple() {
        int[] input = {1, 2, 3, 4, 5};
        int[] expected = {5, 4, 3, 2, 1};
        assertArrayEquals(expected, MathOperations.reverseArray(input));
    }
    
    @Test
    void testReverseArrayEmpty() {
        assertArrayEquals(new int[0], MathOperations.reverseArray(new int[0]));
    }

    @Test
    void testReverseArrayDoesNotMutateInput() {
        int[] input = {7, 8, 9};
        int[] originalCopy = input.clone();
        int[] reversed = MathOperations.reverseArray(input);

        assertArrayEquals(new int[]{9, 8, 7}, reversed);
        assertArrayEquals(originalCopy, input);
        assertNotSame(input, reversed);
    }
    
    @Test
    void testReverseArrayNull() {
        assertNull(MathOperations.reverseArray(null));
    }
    
    // ============ bubbleSort Tests ============
    @Test
    void testBubbleSortSimple() {
        int[] input = {5, 2, 8, 1, 9};
        int[] expected = {1, 2, 5, 8, 9};
        assertArrayEquals(expected, MathOperations.bubbleSort(input));
    }
    
    @Test
    void testBubbleSortAlreadySorted() {
        int[] input = {1, 2, 3, 4, 5};
        int[] expected = {1, 2, 3, 4, 5};
        assertArrayEquals(expected, MathOperations.bubbleSort(input));
    }

    @Test
    void testBubbleSortSingleElement() {
        int[] input = {42};
        int[] expected = {42};
        assertArrayEquals(expected, MathOperations.bubbleSort(input));
    }

    @Test
    void testBubbleSortTwoElements() {
        int[] input = {2, 1};
        int[] expected = {1, 2};
        assertArrayEquals(expected, MathOperations.bubbleSort(input));
    }
    
    @Test
    void testBubbleSortReverse() {
        int[] input = {5, 4, 3, 2, 1};
        int[] expected = {1, 2, 3, 4, 5};
        assertArrayEquals(expected, MathOperations.bubbleSort(input));
    }
    
    @Test
    void testBubbleSortEmpty() {
        assertArrayEquals(new int[0], MathOperations.bubbleSort(new int[0]));
    }
    
    @Test
    void testBubbleSortNull() {
        assertNull(MathOperations.bubbleSort(null));
    }

    @Test
    void testBubbleSortDoesNotMutateInputAndHandlesDuplicates() {
        int[] input = {3, -1, 3, 2, -1};
        int[] originalCopy = input.clone();
        int[] sorted = MathOperations.bubbleSort(input);

        assertArrayEquals(new int[]{-1, -1, 2, 3, 3}, sorted);
        assertArrayEquals(originalCopy, input);
        assertNotSame(input, sorted);
    }
}
