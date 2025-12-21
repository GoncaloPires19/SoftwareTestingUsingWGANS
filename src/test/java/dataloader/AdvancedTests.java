package dataloader;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;
import utils.JUnitCSVLogger;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;

public class AdvancedTests {

    @BeforeEach
    void setUp() {
        // Setup before each test
    }

    // ============== ARRAY OPERATIONS ==============
    @Test
    void testArraySum() {
        int[][] testCases = {
            {1,2,3,6},{5,10,15,30},{2,2,2,6},{10,20,30,60},{1,1,1,3},
            {7,8,9,24},{4,5,6,15},{11,22,33,66},{100,200,300,600},{50,50,50,150}
        };

        for (int[] tc : testCases) {
            int a = tc[0], b = tc[1], c = tc[2], expected = tc[3];
            int result = a + b + c;
            assertEquals(expected, result);
            JUnitCSVLogger.log(Arrays.asList((double)a, (double)b, (double)c, (double)expected));
        }
    }

    @Test
    void testArrayProduct() {
        int[][] testCases = {
            {2,3,4,24},{1,5,10,50},{3,3,3,27},{2,2,5,20},{4,5,6,120},
            {1,1,10,10},{7,2,3,42},{6,6,6,216},{10,2,5,100},{3,4,5,60}
        };

        for (int[] tc : testCases) {
            int a = tc[0], b = tc[1], c = tc[2], expected = tc[3];
            int result = a * b * c;
            assertEquals(expected, result);
            JUnitCSVLogger.log(Arrays.asList((double)a, (double)b, (double)c, (double)expected));
        }
    }

    // ============== SORTING TESTS ==============
    @Test
    void testBubbleSort() {
        int[][] testCases = {
            {5,2,8,1,9},
            {3,1,4,1,5},
            {9,9,9,9,9},
            {1,2,3,4,5},
            {5,4,3,2,1}
        };

        for (int[] arr : testCases) {
            int[] sorted = bubbleSort(arr.clone());
            for (int i = 0; i < sorted.length - 1; i++) {
                assertTrue(sorted[i] <= sorted[i + 1]);
            }
            JUnitCSVLogger.log(Arrays.asList((double)sorted.length));
        }
    }

    // ============== STRING MANIPULATION ==============
    @Test
    void testStringReversal() {
        String[] testCases = {"hello", "world", "a", "racecar", "java"};

        for (String str : testCases) {
            String reversed = new StringBuilder(str).reverse().toString();
            assertEquals(str.length(), reversed.length());
            JUnitCSVLogger.log(Arrays.asList((double)str.length()));
        }
    }

    @Test
    void testStringCounting() {
        String[] testCases = {"hello", "programming", "test", "junit", "automation"};

        for (String str : testCases) {
            int vowels = countVowels(str);
            assertTrue(vowels >= 0 && vowels <= str.length());
            JUnitCSVLogger.log(Arrays.asList((double)str.length(), (double)vowels));
        }
    }

    @Test
    void testStringContains() {
        String[][] testCases = {
            {"hello world", "world", "1"},
            {"java programming", "java", "1"},
            {"test case", "xyz", "0"},
            {"automation testing", "test", "1"},
            {"unit tests", "unit", "1"}
        };

        for (String[] tc : testCases) {
            String haystack = tc[0];
            String needle = tc[1];
            int expected = Integer.parseInt(tc[2]);
            int result = haystack.contains(needle) ? 1 : 0;
            assertEquals(expected, result);
            JUnitCSVLogger.log(Arrays.asList((double)result));
        }
    }

    // ============== MATRIX OPERATIONS ==============
    @Test
    void testMatrixDeterminant2x2() {
        int[][][] testCases = {
            {
                {1, 2},
                {3, 4}
            },
            {
                {5, 7},
                {3, 2}
            },
            {
                {2, 3},
                {4, 5}
            }
        };

        for (int[][] matrix : testCases) {
            int det = determinant2x2(matrix);
            JUnitCSVLogger.log(Arrays.asList((double)det));
        }
    }

    @Test
    void testMatrixTranspose() {
        int[][][] testCases = {
            {
                {1, 2},
                {3, 4}
            },
            {
                {1, 2, 3},
                {4, 5, 6}
            }
        };

        for (int[][] matrix : testCases) {
            int[][] transpose = transposeMatrix(matrix);
            assertEquals(matrix[0].length, transpose.length);
            assertEquals(matrix.length, transpose[0].length);
            JUnitCSVLogger.log(Arrays.asList((double)transpose.length));
        }
    }

    // ============== COMPLEX CALCULATIONS ==============
    @Test
    void testQuadraticFormula() {
        double[][] testCases = {
            {1, -5, 6},    // x^2 - 5x + 6 = 0, roots: 2, 3
            {1, -7, 12},   // x^2 - 7x + 12 = 0, roots: 3, 4
            {1, 0, -4},    // x^2 - 4 = 0, roots: 2, -2
            {2, -8, 6},    // 2x^2 - 8x + 6 = 0, roots: 1, 3
            {1, -6, 9}     // x^2 - 6x + 9 = 0, root: 3
        };

        for (double[] tc : testCases) {
            double a = tc[0], b = tc[1], c = tc[2];
            double discriminant = b * b - 4 * a * c;
            assertTrue(discriminant >= -0.0001);
            JUnitCSVLogger.log(Arrays.asList(a, b, c, discriminant));
        }
    }

    @Test
    void testComplexArithmetic() {
        double[][] testCases = {
            {3, 4, 5},     // sqrt(3^2 + 4^2) = 5
            {5, 12, 13},   // sqrt(5^2 + 12^2) = 13
            {8, 15, 17},   // sqrt(8^2 + 15^2) = 17
            {7, 24, 25},   // sqrt(7^2 + 24^2) = 25
            {6, 8, 10}     // sqrt(6^2 + 8^2) = 10
        };

        for (double[] tc : testCases) {
            double a = tc[0], b = tc[1], expected = tc[2];
            double result = Math.sqrt(a * a + b * b);
            assertEquals(expected, result, 0.01);
            JUnitCSVLogger.log(Arrays.asList(a, b, result));
        }
    }

    // ============== CONDITIONAL LOGIC ==============
    @Test
    void testGradeAssignment() {
        double[][] testCases = {
            {95, 10}, {88, 9}, {78, 8}, {68, 7}, {58, 6},
            {99, 10}, {92, 9}, {85, 9}, {75, 8}, {65, 7}
        };

        for (double[] tc : testCases) {
            double score = tc[0];
            double expectedGrade = tc[1];
            double grade = assignGrade(score);
            assertEquals(expectedGrade, grade);
            JUnitCSVLogger.log(Arrays.asList(score, grade));
        }
    }

    @Test
    void testRangeChecking() {
        int[][] testCases = {
            {5, 1, 10, 1},
            {15, 1, 10, 0},
            {1, 1, 10, 1},
            {10, 1, 10, 1},
            {0, 1, 10, 0},
            {50, 20, 100, 1},
            {150, 20, 100, 0},
            {75, 50, 100, 1},
            {25, 50, 100, 0},
            {100, 0, 100, 1}
        };

        for (int[] tc : testCases) {
            int value = tc[0], min = tc[1], max = tc[2], expected = tc[3];
            int result = (value >= min && value <= max) ? 1 : 0;
            assertEquals(expected, result);
            JUnitCSVLogger.log(Arrays.asList((double)value, (double)min, (double)max, (double)result));
        }
    }

    // ============== LOOP OPERATIONS ==============
    @Test
    void testCountingLoops() {
        int[][] testCases = {
            {5, 15}, {10, 55}, {3, 6}, {7, 28}, {1, 1},
            {8, 36}, {6, 21}, {4, 10}, {9, 45}, {2, 3}
        };

        for (int[] tc : testCases) {
            int n = tc[0];
            int expected = tc[1];
            int result = sumUpTo(n);
            assertEquals(expected, result);
            JUnitCSVLogger.log(Arrays.asList((double)n, (double)result));
        }
    }

    @Test
    void testNestedLoops() {
        int[] testCases = {1, 2, 3, 4, 5};

        for (int n : testCases) {
            int result = countNestedLoopIterations(n);
            assertEquals(n * n, result);
            JUnitCSVLogger.log(Arrays.asList((double)n, (double)result));
        }
    }

    // ============== RECURSION ==============
    @Test
    void testRecursiveFactorial() {
        int[][] testCases = {
            {0, 1}, {1, 1}, {2, 2}, {3, 6}, {4, 24},
            {5, 120}, {6, 720}, {7, 5040}, {8, 40320}, {9, 362880}
        };

        for (int[] tc : testCases) {
            int n = tc[0];
            int expected = tc[1];
            int result = recursiveFactorial(n);
            assertEquals(expected, result);
            JUnitCSVLogger.log(Arrays.asList((double)n, (double)result));
        }
    }

    @Test
    void testRecursiveFibonacci() {
        int[][] testCases = {
            {0, 0}, {1, 1}, {2, 1}, {3, 2}, {4, 3},
            {5, 5}, {6, 8}, {7, 13}, {8, 21}, {9, 34}
        };

        for (int[] tc : testCases) {
            int n = tc[0];
            int expected = tc[1];
            int result = recursiveFibonacci(n);
            assertEquals(expected, result);
            JUnitCSVLogger.log(Arrays.asList((double)n, (double)result));
        }
    }

    // ============== BITWISE COMPLEX OPERATIONS ==============
    @Test
    void testBitCount() {
        int[] testCases = {0, 1, 3, 7, 15, 31, 63, 127, 255, 511};

        for (int num : testCases) {
            int bitCount = Integer.bitCount(num);
            assertTrue(bitCount >= 0);
            JUnitCSVLogger.log(Arrays.asList((double)num, (double)bitCount));
        }
    }

    @Test
    void testPowerOfTwo() {
        int[] testCases = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};

        for (int num : testCases) {
            int result = isPowerOfTwo(num) ? 1 : 0;
            assertEquals(1, result);
            JUnitCSVLogger.log(Arrays.asList((double)num, (double)result));
        }
    }

    // ============== HELPER METHODS ==============

    private int[] bubbleSort(int[] arr) {
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

    private int countVowels(String str) {
        int count = 0;
        for (char c : str.toLowerCase().toCharArray()) {
            if ("aeiou".indexOf(c) >= 0) count++;
        }
        return count;
    }

    private int determinant2x2(int[][] matrix) {
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    }

    private int[][] transposeMatrix(int[][] matrix) {
        int[][] result = new int[matrix[0].length][matrix.length];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                result[j][i] = matrix[i][j];
            }
        }
        return result;
    }

    private double assignGrade(double score) {
        if (score >= 95) return 10;
        if (score >= 85) return 9;
        if (score >= 75) return 8;
        if (score >= 65) return 7;
        if (score >= 55) return 6;
        return 5;
    }

    private int sumUpTo(int n) {
        int sum = 0;
        for (int i = 1; i <= n; i++) {
            sum += i;
        }
        return sum;
    }

    private int countNestedLoopIterations(int n) {
        int count = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                count++;
            }
        }
        return count;
    }

    private int recursiveFactorial(int n) {
        if (n <= 1) return 1;
        return n * recursiveFactorial(n - 1);
    }

    private int recursiveFibonacci(int n) {
        if (n <= 1) return n;
        return recursiveFibonacci(n - 1) + recursiveFibonacci(n - 2);
    }

    private boolean isPowerOfTwo(int n) {
        return n > 0 && (n & (n - 1)) == 0;
    }
}
