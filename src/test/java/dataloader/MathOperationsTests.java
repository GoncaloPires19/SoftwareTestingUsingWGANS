package dataloader;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test suite for MathOperations.
 *
 * Organised with @Nested classes per method to keep failure output readable.
 * Parametrized tests reduce duplication while maintaining branch coverage.
 * Every public method has: happy-path, boundary, and exception-path tests.
 */
@DisplayName("MathOperations")
class MathOperationsTests {

    // =========================================================================
    // sum
    // =========================================================================

    @Nested
    @DisplayName("sum(int[])")
    class SumTests {

        @Test
        @DisplayName("returns 0 for null input")
        void nullArray() {
            assertEquals(0, MathOperations.sum(null));
        }

        @Test
        @DisplayName("returns 0 for empty array")
        void emptyArray() {
            assertEquals(0, MathOperations.sum(new int[0]));
        }

        @Test
        @DisplayName("single element")
        void singleElement() {
            assertEquals(5, MathOperations.sum(new int[]{5}));
        }

        @ParameterizedTest(name = "sum({0}) = {1}")
        @CsvSource({
            "'1,2,3,4,5',    15",
            "'-1,-2,-3,-4,-5', -15",
            "'-5,3,2,-3,3',    0",
            "'0,0,0',          0",
            "'2147483647,-2147483647', 0"
        })
        @DisplayName("parametrized cases")
        void parametrized(String csv, int expected) {
            int[] arr = parseIntArray(csv);
            assertEquals(expected, MathOperations.sum(arr));
        }
    }

    // =========================================================================
    // max
    // =========================================================================

    @Nested
    @DisplayName("max(int[])")
    class MaxTests {

        @Test
        @DisplayName("throws on null")
        void throwsOnNull() {
            assertThrows(IllegalArgumentException.class, () -> MathOperations.max(null));
        }

        @Test
        @DisplayName("throws on empty array")
        void throwsOnEmpty() {
            assertThrows(IllegalArgumentException.class, () -> MathOperations.max(new int[0]));
        }

        @Test
        @DisplayName("single element is its own max")
        void singleElement() {
            assertEquals(42, MathOperations.max(new int[]{42}));
        }

        @ParameterizedTest(name = "max({0}) = {1}")
        @CsvSource({
            "'1,5,9,3,2',    9",
            "'-10,-5,-1,-8', -1",
            "'-50,0,100,25', 100",
            "'7,7,7',        7"
        })
        @DisplayName("parametrized cases")
        void parametrized(String csv, int expected) {
            assertEquals(expected, MathOperations.max(parseIntArray(csv)));
        }
    }

    // =========================================================================
    // min
    // =========================================================================

    @Nested
    @DisplayName("min(int[])")
    class MinTests {

        @Test
        @DisplayName("throws on null")
        void throwsOnNull() {
            assertThrows(IllegalArgumentException.class, () -> MathOperations.min(null));
        }

        @Test
        @DisplayName("throws on empty array")
        void throwsOnEmpty() {
            assertThrows(IllegalArgumentException.class, () -> MathOperations.min(new int[0]));
        }

        @Test
        @DisplayName("single element")
        void singleElement() {
            assertEquals(-7, MathOperations.min(new int[]{-7}));
        }

        @ParameterizedTest(name = "min({0}) = {1}")
        @CsvSource({
            "'1,5,9,3,2',    1",
            "'-10,-5,-1,-8', -10",
            "'-50,0,100,25', -50",
            "'7,7,7',        7"
        })
        @DisplayName("parametrized cases")
        void parametrized(String csv, int expected) {
            assertEquals(expected, MathOperations.min(parseIntArray(csv)));
        }
    }

    // =========================================================================
    // average
    // =========================================================================

    @Nested
    @DisplayName("average(int[])")
    class AverageTests {

        @Test
        @DisplayName("throws on null")
        void throwsOnNull() {
            assertThrows(IllegalArgumentException.class, () -> MathOperations.average(null));
        }

        @Test
        @DisplayName("throws on empty array")
        void throwsOnEmpty() {
            assertThrows(IllegalArgumentException.class, () -> MathOperations.average(new int[0]));
        }

        @Test
        @DisplayName("single element")
        void singleElement() {
            assertEquals(3.0, MathOperations.average(new int[]{3}), 1e-9);
        }

        @Test
        @DisplayName("positive integers")
        void positiveIntegers() {
            assertEquals(3.0, MathOperations.average(new int[]{1, 2, 3, 4, 5}), 1e-9);
        }

        @Test
        @DisplayName("non-integer average")
        void nonIntegerAverage() {
            assertEquals(1.5, MathOperations.average(new int[]{1, 2}), 1e-9);
        }

        @Test
        @DisplayName("negative integers")
        void negativeIntegers() {
            assertEquals(-3.0, MathOperations.average(new int[]{-1, -2, -3, -4, -5}), 1e-9);
        }

        @Test
        @DisplayName("mixed integers summing to zero")
        void mixedSumZero() {
            assertEquals(0.0, MathOperations.average(new int[]{-3, 0, 3}), 1e-9);
        }
    }

    // =========================================================================
    // median
    // =========================================================================

    @Nested
    @DisplayName("median(int[])")
    class MedianTests {

        @Test
        @DisplayName("throws on null")
        void throwsOnNull() {
            assertThrows(IllegalArgumentException.class, () -> MathOperations.median(null));
        }

        @Test
        @DisplayName("throws on empty array")
        void throwsOnEmpty() {
            assertThrows(IllegalArgumentException.class, () -> MathOperations.median(new int[0]));
        }

        @Test
        @DisplayName("single element")
        void singleElement() {
            assertEquals(7.0, MathOperations.median(new int[]{7}), 1e-9);
        }

        @Test
        @DisplayName("odd count — middle element")
        void oddCount() {
            assertEquals(3.0, MathOperations.median(new int[]{5, 1, 3}), 1e-9);
        }

        @Test
        @DisplayName("even count — average of two middle values")
        void evenCount() {
            assertEquals(2.5, MathOperations.median(new int[]{1, 2, 3, 4}), 1e-9);
        }

        @Test
        @DisplayName("does not mutate the input array")
        void doesNotMutateInput() {
            int[] input = {4, 2, 6};
            int[] copy = input.clone();
            MathOperations.median(input);
            assertArrayEquals(copy, input);
        }

        @Test
        @DisplayName("negative numbers")
        void negativeNumbers() {
            assertEquals(-3.0, MathOperations.median(new int[]{-5, -3, -1}), 1e-9);
        }
    }

    // =========================================================================
    // countOccurrences
    // =========================================================================

    @Nested
    @DisplayName("countOccurrences(int[], int)")
    class CountOccurrencesTests {

        @Test
        @DisplayName("returns 0 for null array")
        void nullArray() {
            assertEquals(0, MathOperations.countOccurrences(null, 5));
        }

        @Test
        @DisplayName("returns 0 when value not found")
        void notFound() {
            assertEquals(0, MathOperations.countOccurrences(new int[]{1, 2, 3}, 9));
        }

        @Test
        @DisplayName("counts single occurrence")
        void singleOccurrence() {
            assertEquals(1, MathOperations.countOccurrences(new int[]{1, 2, 3}, 2));
        }

        @Test
        @DisplayName("counts multiple occurrences")
        void multipleOccurrences() {
            assertEquals(3, MathOperations.countOccurrences(new int[]{1, 2, 1, 3, 1}, 1));
        }

        @Test
        @DisplayName("all elements match")
        void allMatch() {
            assertEquals(4, MathOperations.countOccurrences(new int[]{7, 7, 7, 7}, 7));
        }

        @Test
        @DisplayName("counts negative target")
        void negativeTarget() {
            assertEquals(2, MathOperations.countOccurrences(new int[]{-1, 0, -1, 1}, -1));
        }
    }

    // =========================================================================
    // clamp
    // =========================================================================

    @Nested
    @DisplayName("clamp(int, int, int)")
    class ClampTests {

        @Test
        @DisplayName("throws when min > max")
        void throwsWhenMinGtMax() {
            assertThrows(IllegalArgumentException.class, () -> MathOperations.clamp(5, 10, 3));
        }

        @Test
        @DisplayName("returns value when within range")
        void withinRange() {
            assertEquals(5, MathOperations.clamp(5, 0, 10));
        }

        @Test
        @DisplayName("clamps to min when below range")
        void belowRange() {
            assertEquals(0, MathOperations.clamp(-5, 0, 10));
        }

        @Test
        @DisplayName("clamps to max when above range")
        void aboveRange() {
            assertEquals(10, MathOperations.clamp(15, 0, 10));
        }

        @Test
        @DisplayName("returns min when value equals min")
        void equalToMin() {
            assertEquals(3, MathOperations.clamp(3, 3, 7));
        }

        @Test
        @DisplayName("returns max when value equals max")
        void equalToMax() {
            assertEquals(7, MathOperations.clamp(7, 3, 7));
        }

        @Test
        @DisplayName("single-point range (min == max)")
        void singlePointRange() {
            assertEquals(5, MathOperations.clamp(99, 5, 5));
            assertEquals(5, MathOperations.clamp(-99, 5, 5));
        }

        @Test
        @DisplayName("negative range")
        void negativeRange() {
            assertEquals(-3, MathOperations.clamp(0, -10, -3));
            assertEquals(-10, MathOperations.clamp(-20, -10, -3));
            assertEquals(-5, MathOperations.clamp(-5, -10, -3));
        }
    }

    // =========================================================================
    // isPrime
    // =========================================================================

    @Nested
    @DisplayName("isPrime(int)")
    class IsPrimeTests {

        @ParameterizedTest(name = "{0} is prime")
        @ValueSource(ints = {2, 3, 5, 7, 11, 13, 17, 19, 97, 101, 1009})
        @DisplayName("prime numbers")
        void primes(int n) {
            assertTrue(MathOperations.isPrime(n));
        }

        @ParameterizedTest(name = "{0} is not prime")
        @ValueSource(ints = {-5, -1, 0, 1, 4, 6, 8, 9, 10, 25, 49, 100})
        @DisplayName("non-prime numbers")
        void nonPrimes(int n) {
            assertFalse(MathOperations.isPrime(n));
        }

        @Test
        @DisplayName("2 is the only even prime")
        void twoIsEvenPrime() {
            assertTrue(MathOperations.isPrime(2));
            assertFalse(MathOperations.isPrime(4));
        }
    }

    // =========================================================================
    // factorial
    // =========================================================================

    @Nested
    @DisplayName("factorial(int)")
    class FactorialTests {

        @Test
        @DisplayName("throws on negative input")
        void throwsOnNegative() {
            assertThrows(IllegalArgumentException.class, () -> MathOperations.factorial(-1));
        }

        @Test
        @DisplayName("throws on n > 20 (overflow)")
        void throwsOnOverflow() {
            assertThrows(ArithmeticException.class, () -> MathOperations.factorial(21));
        }

        @ParameterizedTest(name = "{0}! = {1}")
        @CsvSource({
            "0,  1",
            "1,  1",
            "2,  2",
            "3,  6",
            "4,  24",
            "5,  120",
            "10, 3628800",
            "12, 479001600",
            "20, 2432902008176640000"
        })
        @DisplayName("parametrized factorial values")
        void parametrized(int n, long expected) {
            assertEquals(expected, MathOperations.factorial(n));
        }
    }

    // =========================================================================
    // fibonacci
    // =========================================================================

    @Nested
    @DisplayName("fibonacci(int)")
    class FibonacciTests {

        @Test
        @DisplayName("throws on negative input")
        void throwsOnNegative() {
            assertThrows(IllegalArgumentException.class, () -> MathOperations.fibonacci(-1));
        }

        @Test
        @DisplayName("throws on n > 92 (overflow)")
        void throwsOnOverflow() {
            assertThrows(ArithmeticException.class, () -> MathOperations.fibonacci(93));
        }

        @ParameterizedTest(name = "fib({0}) = {1}")
        @CsvSource({
            "0,  0",
            "1,  1",
            "2,  1",
            "3,  2",
            "4,  3",
            "5,  5",
            "6,  8",
            "10, 55",
            "20, 6765",
            "46, 1836311903"
        })
        @DisplayName("parametrized fibonacci values")
        void parametrized(int n, long expected) {
            assertEquals(expected, MathOperations.fibonacci(n));
        }
    }

    // =========================================================================
    // gcd
    // =========================================================================

    @Nested
    @DisplayName("gcd(int, int)")
    class GcdTests {

        @ParameterizedTest(name = "gcd({0},{1}) = {2}")
        @CsvSource({
            "9,  6,  3",
            "15, 10, 5",
            "7,  5,  1",
            "17, 13, 1",
            "5,  0,  5",
            "0,  5,  5",
            "54, 24, 6",
            "24, 54, 6"
        })
        @DisplayName("positive inputs")
        void positiveInputs(int a, int b, int expected) {
            assertEquals(expected, MathOperations.gcd(a, b));
        }

        @ParameterizedTest(name = "gcd({0},{1}) = {2} (handles negatives)")
        @CsvSource({
            "-9, 6,  3",
            "9, -6,  3",
            "-9, -6, 3",
            "-15, 10, 5"
        })
        @DisplayName("negative inputs — result is always non-negative")
        void negativeInputs(int a, int b, int expected) {
            assertEquals(expected, MathOperations.gcd(a, b));
        }

        @Test
        @DisplayName("gcd(0, 0) returns 0")
        void bothZero() {
            assertEquals(0, MathOperations.gcd(0, 0));
        }

        @Test
        @DisplayName("gcd of a number with itself is the number")
        void sameNumber() {
            assertEquals(7, MathOperations.gcd(7, 7));
        }
    }

    // =========================================================================
    // isPerfectSquare
    // =========================================================================

    @Nested
    @DisplayName("isPerfectSquare(int)")
    class IsPerfectSquareTests {

        @ParameterizedTest(name = "{0} is a perfect square")
        @ValueSource(ints = {0, 1, 4, 9, 16, 25, 36, 49, 100, 10000})
        @DisplayName("perfect squares")
        void perfectSquares(int n) {
            assertTrue(MathOperations.isPerfectSquare(n));
        }

        @ParameterizedTest(name = "{0} is not a perfect square")
        @ValueSource(ints = {2, 3, 5, 7, 8, 10, 15, 99})
        @DisplayName("non-perfect squares")
        void nonPerfectSquares(int n) {
            assertFalse(MathOperations.isPerfectSquare(n));
        }

        @ParameterizedTest(name = "negative {0} returns false")
        @ValueSource(ints = {-1, -4, -9, Integer.MIN_VALUE})
        @DisplayName("negative inputs always return false")
        void negativeInputs(int n) {
            assertFalse(MathOperations.isPerfectSquare(n));
        }
    }

    // =========================================================================
    // reverseArray
    // =========================================================================

    @Nested
    @DisplayName("reverseArray(int[])")
    class ReverseArrayTests {

        @Test
        @DisplayName("returns null for null input")
        void nullInput() {
            assertNull(MathOperations.reverseArray(null));
        }

        @Test
        @DisplayName("empty array")
        void emptyArray() {
            assertArrayEquals(new int[0], MathOperations.reverseArray(new int[0]));
        }

        @Test
        @DisplayName("single element")
        void singleElement() {
            assertArrayEquals(new int[]{5}, MathOperations.reverseArray(new int[]{5}));
        }

        @Test
        @DisplayName("reverses correctly")
        void reversesCorrectly() {
            assertArrayEquals(new int[]{5, 4, 3, 2, 1},
                    MathOperations.reverseArray(new int[]{1, 2, 3, 4, 5}));
        }

        @Test
        @DisplayName("does not mutate the original array")
        void doesNotMutateInput() {
            int[] input = {7, 8, 9};
            int[] copy = input.clone();
            int[] result = MathOperations.reverseArray(input);
            assertArrayEquals(copy, input, "Input must not be modified");
            assertNotSame(input, result, "Must return a new array");
        }

        @Test
        @DisplayName("double reverse restores original order")
        void doubleReverseIsIdentity() {
            int[] input = {1, 2, 3, 4, 5};
            assertArrayEquals(input, MathOperations.reverseArray(MathOperations.reverseArray(input)));
        }
    }

    // =========================================================================
    // bubbleSort
    // =========================================================================

    @Nested
    @DisplayName("bubbleSort(int[])")
    class BubbleSortTests {

        @Test
        @DisplayName("returns null for null input")
        void nullInput() {
            assertNull(MathOperations.bubbleSort(null));
        }

        @Test
        @DisplayName("empty array")
        void emptyArray() {
            assertArrayEquals(new int[0], MathOperations.bubbleSort(new int[0]));
        }

        @Test
        @DisplayName("single element")
        void singleElement() {
            assertArrayEquals(new int[]{42}, MathOperations.bubbleSort(new int[]{42}));
        }

        @Test
        @DisplayName("two elements out of order")
        void twoElements() {
            assertArrayEquals(new int[]{1, 2}, MathOperations.bubbleSort(new int[]{2, 1}));
        }

        @Test
        @DisplayName("already sorted")
        void alreadySorted() {
            assertArrayEquals(new int[]{1, 2, 3, 4, 5},
                    MathOperations.bubbleSort(new int[]{1, 2, 3, 4, 5}));
        }

        @Test
        @DisplayName("reverse order")
        void reverseOrder() {
            assertArrayEquals(new int[]{1, 2, 3, 4, 5},
                    MathOperations.bubbleSort(new int[]{5, 4, 3, 2, 1}));
        }

        @Test
        @DisplayName("handles duplicates and does not mutate input")
        void duplicatesAndNoMutation() {
            int[] input = {3, -1, 3, 2, -1};
            int[] copy = input.clone();
            int[] sorted = MathOperations.bubbleSort(input);
            assertArrayEquals(new int[]{-1, -1, 2, 3, 3}, sorted);
            assertArrayEquals(copy, input, "Input must not be modified");
            assertNotSame(input, sorted, "Must return a new array");
        }
    }

    // =========================================================================
    // isPalindrome
    // =========================================================================

    @Nested
    @DisplayName("isPalindrome(String)")
    class IsPalindromeTests {

        @ParameterizedTest(name = "\"{0}\" is a palindrome")
        @ValueSource(strings = {"aba", "racecar", "a", "madam", "level"})
        @DisplayName("simple palindromes")
        void simplePalindromes(String s) {
            assertTrue(MathOperations.isPalindrome(s));
        }

        @ParameterizedTest(name = "\"{0}\" is not a palindrome")
        @ValueSource(strings = {"hello", "world", "Panama canal"})
        @DisplayName("non-palindromes")
        void nonPalindromes(String s) {
            assertFalse(MathOperations.isPalindrome(s));
        }

        @Test
        @DisplayName("null is a palindrome")
        void nullIsPalindrome() {
            assertTrue(MathOperations.isPalindrome(null));
        }

        @Test
        @DisplayName("empty string is a palindrome")
        void emptyIsPalindrome() {
            assertTrue(MathOperations.isPalindrome(""));
        }

        @Test
        @DisplayName("ignores spaces")
        void ignoresSpaces() {
            assertTrue(MathOperations.isPalindrome("a b a"));
            assertTrue(MathOperations.isPalindrome("race car"));
        }

        @Test
        @DisplayName("ignores case and punctuation")
        void ignoresCaseAndPunctuation() {
            assertTrue(MathOperations.isPalindrome("A man, a plan, a canal: Panama!"));
            assertTrue(MathOperations.isPalindrome("No 'x' in Nixon"));
        }
    }

    // =========================================================================
    // Helper
    // =========================================================================

    private static int[] parseIntArray(String csv) {
        String[] parts = csv.split(",");
        int[] arr = new int[parts.length];
        for (int i = 0; i < parts.length; i++) {
            arr[i] = Integer.parseInt(parts[i].trim());
        }
        return arr;
    }
}
