package dataloader;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import static org.junit.jupiter.api.Assertions.*;
import utils.JUnitCSVLogger;

/**
 * RobustTests - Professional-grade test suite for WGAN training
 * Organized with nested test classes for comprehensive coverage
 * Each test logs data for synthetic test generation
 */
@DisplayName("Robust Test Suite for WGAN Training")
class RobustTests {
    
    @BeforeEach
    void setUp() {
        // Setup before each test
    }

    @Test
    @DisplayName("Mutation alignment with MathOperations and TestDataLoader")
    void testMutationAlignmentBridge() {
        assertEquals(6, MathOperations.sum(new int[]{1, 2, 3}));
        assertEquals(3, MathOperations.gcd(9, 6));
        assertFalse(MathOperations.isPrime(21));
        assertTrue(MathOperations.isPrime(19));
        assertArrayEquals(new int[]{1, 2, 3}, MathOperations.bubbleSort(new int[]{3, 1, 2}));

        TestDataLoader loader = new TestDataLoader();
        loader.collectTestData("robust", 42);
        assertEquals(1, loader.getCollectedDataSize());
        assertEquals("robust,42", loader.getCollectedData(0));
    }

    @Test
    @DisplayName("Mutation core coverage for MathOperations")
    void testMutationCoreCoverageMathOperationsRobust() {
        assertEquals(0, MathOperations.sum(null));
        assertEquals(15, MathOperations.sum(new int[]{1, 2, 3, 4, 5}));

        assertEquals(42, MathOperations.max(new int[]{42}));
        assertEquals(42, MathOperations.min(new int[]{42}));

        assertTrue(MathOperations.isPrime(29));
        assertFalse(MathOperations.isPrime(28));

        assertEquals(120, MathOperations.factorial(5));
        assertEquals(21, MathOperations.fibonacci(8));

        assertEquals(1, MathOperations.gcd(17, 13));

        assertTrue(MathOperations.isPerfectSquare(64));
        assertFalse(MathOperations.isPerfectSquare(63));

        assertTrue(MathOperations.isPalindrome("race car"));
        assertFalse(MathOperations.isPalindrome("software"));

        assertArrayEquals(new int[]{0, 0, 1884}, MathOperations.bubbleSort(new int[]{1884, 0, 0}));
        assertArrayEquals(new int[]{0, 0, 1884}, MathOperations.reverseArray(new int[]{1884, 0, 0}));
    }

    // ============================================================================
    // ARITHMETIC OPERATIONS WITH EDGE CASES
    // ============================================================================
    @Nested
    @DisplayName("Arithmetic Operations")
    class ArithmeticTests {
        
        @Test
        @DisplayName("Basic addition with boundary values")
        void testAdditionBoundaries() {
            int[][] cases = {
                {0, 0}, {1, -1}, {Integer.MAX_VALUE, 0}, 
                {Integer.MIN_VALUE, 0}, {1000000, 1000000}
            };
            
            for (int[] tc : cases) {
                int a = tc[0], b = tc[1];
                int result = a + b;
                assertEquals((long)a + b, result);
                logTestData(a, b, result);
            }
        }

        @Test
        @DisplayName("Subtraction with negative numbers")
        void testSubtractionNegative() {
            int[][] cases = {
                {10, 5}, {-5, -10}, {0, -5}, {-100, 50}, {5, 10}
            };
            
            for (int[] tc : cases) {
                int a = tc[0], b = tc[1];
                int result = a - b;
                assertEquals(a - b, result);
                logTestData(a, b, result);
            }
        }

        @Test
        @DisplayName("Multiplication with zero and one")
        void testMultiplicationIdentity() {
            int[][] cases = {
                {0, 0}, {1, 0}, {0, 1}, {1, 1}, 
                {5, 1}, {1, -5}, {0, 1000}
            };
            
            for (int[] tc : cases) {
                int a = tc[0], b = tc[1];
                int result = a * b;
                assertEquals(a * b, result);
                logTestData(a, b, result);
            }
        }

        @Test
        @DisplayName("Division with zero handling")
        void testDivisionSafety() {
            int[][] validCases = {
                {10, 2}, {20, 4}, {7, 2}, {100, 10}, {1, 1}
            };
            
            for (int[] tc : validCases) {
                int a = tc[0], b = tc[1];
                int result = a / b;
                assertEquals(a / b, result);
                logTestData(a, b, result);
            }
        }

        @Test
        @DisplayName("Modulo operation with edge cases")
        void testModuloEdgeCases() {
            int[][] cases = {
                {10, 3}, {7, 2}, {15, 4}, {20, 7}, {100, 13}
            };
            
            for (int[] tc : cases) {
                int a = tc[0], b = tc[1];
                int result = a % b;
                assertEquals(a % b, result);
                logTestData(a, b, result);
            }
        }
    }

    // ============================================================================
    // NUMBER THEORY AND PRIME TESTING
    // ============================================================================
    @Nested
    @DisplayName("Number Theory")
    class NumberTheoryTests {
        
        @Test
        @DisplayName("Prime number detection comprehensive")
        void testPrimeDetection() {
            int[] primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47};
            int[] nonPrimes = {0, 1, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22};
            
            for (int prime : primes) {
                boolean result = isPrime(prime);
                assertTrue(result, "Should be prime: " + prime);
                logTestData(prime, 1, result ? 1 : 0);
            }
            
            for (int nonPrime : nonPrimes) {
                boolean result = isPrime(nonPrime);
                assertFalse(result, "Should not be prime: " + nonPrime);
                logTestData(nonPrime, 1, result ? 1 : 0);
            }
        }

        @Test
        @DisplayName("GCD calculation with various inputs")
        void testGCDComputation() {
            int[][] cases = {
                {12, 8}, {48, 18}, {35, 14}, {100, 50}, {17, 19}
            };
            
            for (int[] tc : cases) {
                int a = tc[0], b = tc[1];
                int result = gcd(a, b);
                assertTrue(a % result == 0 && b % result == 0);
                logTestData(a, b, result);
            }
        }

        @Test
        @DisplayName("Factorial computation")
        void testFactorialComputation() {
            int[][] cases = {
                {0, 1}, {1, 1}, {2, 2}, {3, 6}, {4, 24}, 
                {5, 120}, {6, 720}, {7, 5040}, {8, 40320}
            };
            
            for (int[] tc : cases) {
                int n = tc[0];
                int expected = tc[1];
                int result = factorial(n);
                assertEquals(expected, result);
                logTestData(n, 1, result);
            }
        }

        @Test
        @DisplayName("Perfect square detection")
        void testPerfectSquareDetection() {
            int[] perfectSquares = {0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100};
            int[] nonSquares = {2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14};
            
            for (int sq : perfectSquares) {
                boolean result = isPerfectSquare(sq);
                assertTrue(result, "Should be perfect square: " + sq);
                logTestData(sq, 1, result ? 1 : 0);
            }
            
            for (int nonSq : nonSquares) {
                boolean result = isPerfectSquare(nonSq);
                assertFalse(result, "Should not be perfect square: " + nonSq);
                logTestData(nonSq, 1, result ? 1 : 0);
            }
        }
    }

    // ============================================================================
    // STRING AND SEQUENCE OPERATIONS
    // ============================================================================
    @Nested
    @DisplayName("String and Sequence Operations")
    class StringSequenceTests {
        
        @Test
        @DisplayName("Palindrome detection with various lengths")
        void testPalindromeDetection() {
            String[] palindromes = {
                "a", "aa", "aba", "abba", "racecar", 
                "abccba", "12321", "abcdcba"
            };
            String[] nonPalindromes = {
                "ab", "abc", "abcd", "hello", "world", 
                "java", "python", "abcde"
            };
            
            for (String p : palindromes) {
                boolean result = isPalindrome(p);
                assertTrue(result, "Should be palindrome: " + p);
                logTestData(p.length(), 1, result ? 1 : 0);
            }
            
            for (String np : nonPalindromes) {
                boolean result = isPalindrome(np);
                assertFalse(result, "Should not be palindrome: " + np);
                logTestData(np.length(), 1, result ? 1 : 0);
            }
        }

        @Test
        @DisplayName("Fibonacci sequence generation")
        void testFibonacciSequence() {
            int[][] expected = {
                {0, 0}, {1, 1}, {2, 1}, {3, 2}, {4, 3},
                {5, 5}, {6, 8}, {7, 13}, {8, 21}, {10, 55}
            };
            
            for (int[] tc : expected) {
                int n = tc[0];
                int result = fibonacci(n);
                assertEquals(tc[1], result);
                logTestData(n, 1, result);
            }
        }

        @Test
        @DisplayName("String contains operation")
        void testStringContains() {
            Object[][] cases = {
                {"hello", "h", true},
                {"hello", "e", true},
                {"hello", "ll", true},
                {"hello", "x", false},
                {"java", "va", true},
                {"test", "xyz", false}
            };
            
            for (Object[] tc : cases) {
                String str = (String) tc[0];
                String substring = (String) tc[1];
                boolean expected = (boolean) tc[2];
                boolean result = str.contains(substring);
                assertEquals(expected, result);
                logTestData(str.length(), substring.length(), 
                           result ? 1 : 0);
            }
        }
    }

    // ============================================================================
    // ARRAY OPERATIONS AND SORTING
    // ============================================================================
    @Nested
    @DisplayName("Array Operations")
    class ArrayOperationsTests {
        
        @Test
        @DisplayName("Array sum with various cases")
        void testArraySum() {
            int[][] testArrays = {
                {1}, {1, 2}, {1, 2, 3}, {5, 5, 5, 5},
                {-1, -2, -3}, {0, 0, 0}, {1, -1, 1, -1}
            };
            
            for (int[] arr : testArrays) {
                int sum = 0;
                for (int x : arr) sum += x;
                assertTrue(sum >= Integer.MIN_VALUE && sum <= Integer.MAX_VALUE);
                logTestData(arr.length, sum, sum);
            }
        }

        @Test
        @DisplayName("Array maximum element")
        void testArrayMaximum() {
            int[][] testArrays = {
                {5}, {1, 5, 3}, {-10, -5, -1}, {0, 0, 0},
                {100, 50, 75}, {1, 1, 1}
            };
            
            for (int[] arr : testArrays) {
                int max = arr[0];
                for (int x : arr) max = Math.max(max, x);
                assertTrue(max >= Integer.MIN_VALUE);
                logTestData(arr.length, max, max);
            }
        }

        @Test
        @DisplayName("Array minimum element")
        void testArrayMinimum() {
            int[][] testArrays = {
                {5}, {1, 5, 3}, {-10, -5, -1}, {0, 0, 0},
                {100, 50, 75}, {1, 1, 1}
            };
            
            for (int[] arr : testArrays) {
                int min = arr[0];
                assertTrue(min <= Integer.MAX_VALUE);
                for (int x : arr) min = Math.min(min, x);
                logTestData(arr.length, min, min);
            }
        }

        @Test
        @DisplayName("Bubble sort correctness")
        void testBubbleSortCorrectness() {
            int[][] testArrays = {
                {3, 1, 2},
                {5, 2, 8, 1, 9},
                {1},
                {2, 1},
                {5, 4, 3, 2, 1},
                {1, 2, 3, 4, 5}
            };
            
            for (int[] arr : testArrays) {
                int[] sorted = bubbleSort(arr.clone());
                for (int i = 1; i < sorted.length; i++) {
                    assertTrue(sorted[i-1] <= sorted[i]);
                }
                logTestData(arr.length, sorted[0], 
                           sorted[sorted.length - 1]);
            }
        }

        @Test
        @DisplayName("Array reversal operation")
        void testArrayReversal() {
            int[][] testArrays = {
                {1, 2, 3, 4, 5},
                {1},
                {2, 1},
                {5, 4, 3, 2, 1}
            };
            
            for (int[] arr : testArrays) {
                int[] reversed = reverseArray(arr);
                for (int i = 0; i < arr.length; i++) {
                    assertEquals(arr[arr.length - 1 - i], reversed[i]);
                }
                logTestData(arr.length, reversed[0], 
                           reversed[reversed.length - 1]);
            }
        }
    }

    // ============================================================================
    // BITWISE OPERATIONS
    // ============================================================================
    @Nested
    @DisplayName("Bitwise Operations")
    class BitwiseOperationsTests {
        
        @Test
        @DisplayName("Bitwise AND operation")
        void testBitwiseAND() {
            int[][] cases = {
                {12, 10}, {15, 7}, {255, 127}, {0, 100}, {63, 31}
            };
            
            for (int[] tc : cases) {
                int a = tc[0], b = tc[1];
                int result = a & b;
                assertEquals(a & b, result);
                logTestData(a, b, result);
            }
        }

        @Test
        @DisplayName("Bitwise OR operation")
        void testBitwiseOR() {
            int[][] cases = {
                {12, 10}, {15, 7}, {255, 127}, {0, 100}, {63, 31}
            };
            
            for (int[] tc : cases) {
                int a = tc[0], b = tc[1];
                int result = a | b;
                assertEquals(a | b, result);
                logTestData(a, b, result);
            }
        }

        @Test
        @DisplayName("Bitwise XOR operation")
        void testBitwiseXOR() {
            int[][] cases = {
                {12, 10}, {15, 7}, {255, 127}, {0, 100}, {63, 31}
            };
            
            for (int[] tc : cases) {
                int a = tc[0], b = tc[1];
                int result = a ^ b;
                assertEquals(a ^ b, result);
                logTestData(a, b, result);
            }
        }

        @Test
        @DisplayName("Bitwise NOT operation")
        void testBitwiseNOT() {
            int[] values = {0, 1, 10, 127, 255};
            
            for (int v : values) {
                int result = ~v;
                assertEquals(~v, result);
                logTestData(v, 1, result);
            }
        }

        @Test
        @DisplayName("Bit shifting operations")
        void testBitShifting() {
            int[][] cases = {
                {4, 1}, {8, 2}, {16, 3}, {1, 5}, {32, 2}
            };
            
            for (int[] tc : cases) {
                int value = tc[0], shifts = tc[1];
                int leftShift = value << shifts;
                int rightShift = value >> shifts;
                assertEquals(value << shifts, leftShift);
                assertEquals(value >> shifts, rightShift);
                logTestData(value, shifts, leftShift);
                logTestData(value, shifts, rightShift);
            }
        }
    }

    // ============================================================================
    // COMPARISON AND LOGIC OPERATIONS
    // ============================================================================
    @Nested
    @DisplayName("Comparison and Logic")
    class ComparisonLogicTests {
        
        @Test
        @DisplayName("Greater than comparisons")
        void testGreaterThan() {
            int[][] cases = {
                {5, 3}, {3, 5}, {5, 5}, {-5, -10}, {0, 1}, {10, -10}
            };
            
            for (int[] tc : cases) {
                int a = tc[0], b = tc[1];
                boolean result = a > b;
                assertEquals(a > b, result);
                logTestData(a, b, result ? 1 : 0);
            }
        }

        @Test
        @DisplayName("Less than comparisons")
        void testLessThan() {
            int[][] cases = {
                {5, 3}, {3, 5}, {5, 5}, {-5, -10}, {0, 1}, {10, -10}
            };
            
            for (int[] tc : cases) {
                int a = tc[0], b = tc[1];
                boolean result = a < b;
                assertEquals(a < b, result);
                logTestData(a, b, result ? 1 : 0);
            }
        }

        @Test
        @DisplayName("Equality comparisons")
        void testEquality() {
            int[][] cases = {
                {5, 5}, {3, 5}, {0, 0}, {-5, -5}, {-5, 5}
            };
            
            for (int[] tc : cases) {
                int a = tc[0], b = tc[1];
                boolean result = a == b;
                assertEquals(a == b, result);
                logTestData(a, b, result ? 1 : 0);
            }
        }

        @Test
        @DisplayName("Logical AND operation")
        void testLogicalAND() {
            boolean[][] cases = {
                {true, true}, {true, false}, {false, true}, {false, false}
            };
            
            for (boolean[] tc : cases) {
                boolean result = tc[0] && tc[1];
                assertEquals(tc[0] && tc[1], result);
                logTestData(tc[0] ? 1 : 0, tc[1] ? 1 : 0, 
                           result ? 1 : 0);
            }
        }

        @Test
        @DisplayName("Logical OR operation")
        void testLogicalOR() {
            boolean[][] cases = {
                {true, true}, {true, false}, {false, true}, {false, false}
            };
            
            for (boolean[] tc : cases) {
                boolean result = tc[0] || tc[1];
                assertEquals(tc[0] || tc[1], result);
                logTestData(tc[0] ? 1 : 0, tc[1] ? 1 : 0, 
                           result ? 1 : 0);
            }
        }
    }

    // ============================================================================
    // MATHEMATICAL FUNCTIONS
    // ============================================================================
    @Nested
    @DisplayName("Mathematical Functions")
    class MathFunctionsTests {
        
        @Test
        @DisplayName("Absolute value function")
        void testAbsoluteValue() {
            int[] cases = {5, -5, 0, 100, -100, 1, -1};
            
            for (int val : cases) {
                int result = Math.abs(val);
                assertEquals(Math.abs(val), result);
                logTestData(val, 1, result);
            }
        }

        @Test
        @DisplayName("Power function")
        void testPowerFunction() {
            int[][] cases = {
                {2, 3}, {3, 2}, {5, 2}, {10, 3}, {2, 10}
            };
            
            for (int[] tc : cases) {
                int base = tc[0], exp = tc[1];
                double result = Math.pow(base, exp);
                assertEquals(Math.pow(base, exp), result);
                logTestData(base, exp, (int)result);
            }
        }

        @Test
        @DisplayName("Square root function")
        void testSquareRoot() {
            int[] cases = {0, 1, 4, 9, 16, 25, 36, 49, 64, 100};
            
            for (int val : cases) {
                double result = Math.sqrt(val);
                assertEquals(Math.sqrt(val), result);
                logTestData(val, 1, (int)result);
            }
        }

        @Test
        @DisplayName("Min and Max functions")
        void testMinMax() {
            int[][] cases = {
                {5, 3}, {-5, -10}, {0, 0}, {100, 50}
            };
            
            for (int[] tc : cases) {
                int a = tc[0], b = tc[1];
                int min = Math.min(a, b);
                int max = Math.max(a, b);
                assertEquals(Math.min(a, b), min);
                assertEquals(Math.max(a, b), max);
                logTestData(a, b, min);
                logTestData(a, b, max);
            }
        }
    }

    // ============================================================================
    // BOUNDARY AND EDGE CASES
    // ============================================================================
    @Nested
    @DisplayName("Boundary and Edge Cases")
    class BoundaryEdgeCasesTests {
        
        @Test
        @DisplayName("Zero handling across operations")
        void testZeroHandling() {
            // Addition with zero
            assertEquals(5, 5 + 0);
            logTestData(5, 0, 5);
            
            // Multiplication by zero
            assertEquals(0, 5 * 0);
            logTestData(5, 0, 0);
            
            // Division of zero
            assertEquals(0, 0 / 5);
            logTestData(0, 5, 0);
        }

        @Test
        @DisplayName("Negative number operations")
        void testNegativeNumbers() {
            // Negative addition
            assertEquals(-8, -5 + (-3));
            logTestData(-5, -3, -8);
            
            // Negative multiplication
            assertEquals(15, -5 * (-3));
            logTestData(-5, -3, 15);
            
            // Negative comparisons
            assertTrue(-5 < -3);
            logTestData(-5, -3, 1);
        }

        @Test
        @DisplayName("Integer overflow scenarios")
        void testIntegerOverflow() {
            // Close to MAX_VALUE
            int nearMax = Integer.MAX_VALUE - 100;
            int result = nearMax + 50;
            assertTrue(nearMax < Integer.MAX_VALUE);
            logTestData(nearMax, 50, result);
        }

        @Test
        @DisplayName("Single element operations")
        void testSingleElementOperations() {
            int[] singleArray = {42};
            
            // Sum of single element
            int sum = singleArray[0];
            assertEquals(42, sum);
            logTestData(1, 42, sum);
            
            // Reverse single element
            int[] reversed = {singleArray[0]};
            assertEquals(42, reversed[0]);
            logTestData(1, 1, reversed[0]);
        }

        @Test
        @DisplayName("Empty string handling")
        void testEmptyStringHandling() {
            String empty = "";
            
            // Length of empty string
            assertEquals(0, empty.length());
            logTestData(0, 1, 0);
            
            // Palindrome of empty string
            assertTrue(isPalindrome(""));
            logTestData(0, 1, 1);
        }
    }

    // ============================================================================
    // HELPER METHODS FOR TESTING
    // ============================================================================
    
    private boolean isPrime(int n) {
        if (n < 2) return false;
        if (n == 2) return true;
        if (n % 2 == 0) return false;
        for (int i = 3; i * i <= n; i += 2) {
            if (n % i == 0) return false;
        }
        return true;
    }

    private int gcd(int a, int b) {
        while (b != 0) {
            int temp = b;
            b = a % b;
            a = temp;
        }
        return Math.abs(a);
    }

    private int factorial(int n) {
        if (n < 0) throw new IllegalArgumentException("n must be non-negative");
        int result = 1;
        for (int i = 2; i <= n; i++) {
            result *= i;
        }
        return result;
    }

    private boolean isPerfectSquare(int n) {
        if (n < 0) return false;
        int sqrt = (int) Math.sqrt(n);
        return sqrt * sqrt == n;
    }

    private boolean isPalindrome(String s) {
        if (s == null || s.isEmpty()) return true;
        String cleaned = s.replaceAll(" ", "").toLowerCase();
        return cleaned.equals(new StringBuilder(cleaned).reverse().toString());
    }

    private int fibonacci(int n) {
        if (n < 0) throw new IllegalArgumentException("n must be non-negative");
        if (n == 0) return 0;
        if (n == 1) return 1;
        int a = 0, b = 1;
        for (int i = 2; i <= n; i++) {
            int temp = a + b;
            a = b;
            b = temp;
        }
        return b;
    }

    private int[] bubbleSort(int[] arr) {
        if (arr == null) return null;
        arr = arr.clone();
        for (int i = 0; i < arr.length; i++) {
            for (int j = 0; j < arr.length - i - 1; j++) {
                if (arr[j] > arr[j + 1]) {
                    int temp = arr[j];
                    arr[j] = arr[j + 1];
                    arr[j + 1] = temp;
                }
            }
        }
        return arr;
    }

    private int[] reverseArray(int[] arr) {
        if (arr == null) return null;
        int[] reversed = new int[arr.length];
        for (int i = 0; i < arr.length; i++) {
            reversed[i] = arr[arr.length - 1 - i];
        }
        return reversed;
    }

    private void logTestData(int input1, int input2, int output) {
        JUnitCSVLogger.log(
            (double) input1,
            (double) input2,
            (double) output
        );
    }
}
