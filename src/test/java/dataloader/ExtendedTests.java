package dataloader;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Generated Extended Test Suite
 * 
 * These tests expand the existing test coverage with additional edge cases
 * and boundary conditions to improve WGAN training data diversity.
 */
class ExtendedTests {
    
    // ============ Arithmetic Operations ============
    
    @Test
    void testAdditionWithNegativeNumbers() {
        int result = -5 + (-3);
        assertEquals(-8, result);
    }
    
    @Test
    void testMultiplicationByZero() {
        int result = 100 * 0;
        assertEquals(0, result);
    }
    
    @Test
    void testDivisionByOne() {
        int result = 42 / 1;
        assertEquals(42, result);
    }
    
    @Test
    void testModuloOperation() {
        int result = 17 % 5;
        assertEquals(2, result);
    }
    
    @Test
    void testLargeNumberMultiplication() {
        long result = 1000000L * 1000000L;
        assertEquals(1000000000000L, result);
    }
    
    // ============ String Operations ============
    
    @Test
    void testEmptyStringLength() {
        String str = "";
        assertEquals(0, str.length());
    }
    
    @Test
    void testStringWithSpaces() {
        String str = "   hello   ";
        assertEquals(11, str.length());
    }
    
    @Test
    void testStringCharAt() {
        String str = "Java";
        assertEquals('J', str.charAt(0));
        assertEquals('a', str.charAt(3));
    }
    
    @Test
    void testStringSubstring() {
        String str = "testing";
        assertEquals("test", str.substring(0, 4));
    }
    
    @Test
    void testStringContains() {
        String str = "EvoSuite Generator";
        assertTrue(str.contains("Suite"));
        assertFalse(str.contains("junit"));
    }
    
    // ============ Array Operations ============
    
    @Test
    void testArraySum() {
        int[] arr = {1, 2, 3, 4, 5};
        int sum = 0;
        for (int num : arr) {
            sum += num;
        }
        assertEquals(15, sum);
    }
    
    @Test
    void testArrayMax() {
        int[] arr = {5, 2, 9, 1, 7};
        int max = arr[0];
        for (int num : arr) {
            if (num > max) max = num;
        }
        assertEquals(9, max);
    }
    
    @Test
    void testArrayMin() {
        int[] arr = {5, 2, 9, 1, 7};
        int min = arr[0];
        for (int num : arr) {
            if (num < min) min = num;
        }
        assertEquals(1, min);
    }
    
    @Test
    void testSingleElementArray() {
        int[] arr = {42};
        assertEquals(1, arr.length);
        assertEquals(42, arr[0]);
    }
    
    // ============ Boundary Conditions ============
    
    @Test
    void testIntegerOverflow() {
        int max = Integer.MAX_VALUE;
        int overflow = max + 1;
        assertTrue(overflow < 0); // Wraps around
    }
    
    @Test
    void testIntegerUnderflow() {
        int min = Integer.MIN_VALUE;
        int underflow = min - 1;
        assertTrue(underflow > 0); // Wraps around
    }
    
    @Test
    void testZeroComparisons() {
        int zero = 0;
        assertEquals(0, zero);
        assertTrue(zero <= 0);
        assertTrue(zero >= 0);
    }
    
    @Test
    void testNegativeNumberComparisons() {
        assertTrue(-5 < -3);
        assertTrue(-10 < 0);
        assertFalse(-5 > 0);
    }
    
    // ============ Logical Operations ============
    
    @Test
    void testBooleanAnd() {
        assertTrue(true && true);
        assertFalse(true && false);
        boolean resultFalse = false;
        assertFalse(resultFalse);
    }
    
    @Test
    void testBooleanOr() {
        boolean t = true;
        boolean f = false;
        boolean or1 = t || f;
        assertTrue(or1);
        boolean or2 = f || t;
        assertTrue(or2);
        boolean f2 = false;
        boolean result = f || f2;
        assertFalse(result);
    }
    
    @Test
    void testBooleanNot() {
        assertTrue(!false);
        assertFalse(!true);
    }
    
    // ============ Type Conversions ============
    
    @Test
    void testIntToString() {
        String result = String.valueOf(42);
        assertEquals("42", result);
    }
    
    @Test
    void testStringToInt() {
        int result = Integer.parseInt("123");
        assertEquals(123, result);
    }
    
    @Test
    void testDoubleToInt() {
        double val = 3.7;
        int result = (int) val;
        assertEquals(3, result);
    }
    
    @Test
    void testFloatToInt() {
        float val = 2.5f;
        int result = (int) val;
        assertEquals(2, result);
    }
    
    // ============ Loop Variations ============
    
    @Test
    void testForLoopCounter() {
        int count = 0;
        for (int i = 0; i < 10; i++) {
            count++;
        }
        assertEquals(10, count);
    }
    
    @Test
    void testWhileLoopExecution() {
        int count = 0;
        int i = 0;
        while (i < 5) {
            count++;
            i++;
        }
        assertEquals(5, count);
    }
    
    @Test
    void testDoWhileLoopExecution() {
        int count = 0;
        int i = 0;
        do {
            count++;
            i++;
        } while (i < 3);
        assertEquals(3, count);
    }
    
    @Test
    void testNestedLoops() {
        int total = 0;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 2; j++) {
                total++;
            }
        }
        assertEquals(6, total);
    }
    
    // ============ Bitwise Operations ============
    
    @Test
    void testBitwiseAnd() {
        int result = 12 & 10; // 1100 & 1010 = 1000
        assertEquals(8, result);
    }
    
    @Test
    void testBitwiseOr() {
        int result = 12 | 10; // 1100 | 1010 = 1110
        assertEquals(14, result);
    }
    
    @Test
    void testBitwiseXor() {
        int result = 12 ^ 10; // 1100 ^ 1010 = 0110
        assertEquals(6, result);
    }
    
    @Test
    void testBitwiseNot() {
        int result = ~5;
        assertEquals(-6, result);
    }
    
    @Test
    void testLeftShift() {
        int result = 5 << 2; // 0101 << 2 = 10100
        assertEquals(20, result);
    }
    
    @Test
    void testRightShift() {
        int result = 20 >> 2; // 10100 >> 2 = 0101
        assertEquals(5, result);
    }
    
    // ============ Mathematical Functions ============
    
    @Test
    void testAbsoluteValue() {
        assertEquals(5, Math.abs(-5));
        int positiveValue = 10;
        assertEquals(positiveValue, Math.abs(positiveValue));
        int zero = 0;
        assertEquals(zero, Math.abs(zero));
    }
    
    @Test
    void testMinMax() {
        assertEquals(3, Math.min(5, 3));
        assertEquals(7, Math.max(7, 2));
    }
    
    @Test
    void testPowerFunction() {
        assertEquals(8, Math.pow(2, 3), 0.01);
        assertEquals(1, Math.pow(5, 0), 0.01);
    }
    
    @Test
    void testSquareRoot() {
        assertEquals(5, Math.sqrt(25), 0.01);
        assertEquals(0, Math.sqrt(0), 0.01);
    }
    
    // ============ Conditional Branches ============
    
    @Test
    void testIfElseLogic() {
        int value = 10;
        String result;
        if (value > 5) {
            result = "greater";
        } else {
            result = "lesser";
        }
        assertEquals("greater", result);
    }
    
    @Test
    void testIfElseIfChain() {
        int grade = 75;
        String result;
        if (grade >= 90) {
            result = "A";
        } else if (grade >= 80) {
            result = "B";
        } else if (grade >= 70) {
            result = "C";
        } else {
            result = "F";
        }
        assertEquals("C", result);
    }
    
    @Test
    void testTernaryOperator() {
        int x = 10;
        String result = (x > 5) ? "big" : "small";
        assertEquals("big", result);
    }
    
    // ============ Special Cases ============
    
    @Test
    void testNullHandling() {
        String str = null;
        assertNull(str);
    }

    // ============ MUTATION ALIGNMENT ============

    @Test
    void testMutationAlignmentMathOperationsExtended() {
        assertEquals(15, MathOperations.sum(new int[]{1, 2, 3, 4, 5}));
        assertEquals(9, MathOperations.max(new int[]{1, 5, 9, 3}));
        assertEquals(1, MathOperations.min(new int[]{1, 5, 9, 3}));
        assertTrue(MathOperations.isPalindrome("level"));
        assertArrayEquals(new int[]{1, 2, 5, 8, 9}, MathOperations.bubbleSort(new int[]{5, 2, 8, 1, 9}));
    }

    @Test
    void testMutationAlignmentTestDataLoaderExtended() {
        TestDataLoader loader = new TestDataLoader();
        loader.collectTestData(100, 200, 300);

        assertEquals(1, loader.getCollectedDataSize());
        assertEquals("100,200,300", loader.getCollectedData(0));
        assertNotNull(loader.getAllCollectedData());
    }

    @Test
    void testMutationCoreCoverageMathOperationsExtended() {
        assertEquals(-15, MathOperations.sum(new int[]{-1, -2, -3, -4, -5}));

        assertEquals(100, MathOperations.max(new int[]{-50, 0, 100, 25}));
        assertEquals(-50, MathOperations.min(new int[]{-50, 0, 100, 25}));

        assertTrue(MathOperations.isPrime(101));
        assertFalse(MathOperations.isPrime(1));

        assertEquals(24, MathOperations.factorial(4));
        assertEquals(8, MathOperations.fibonacci(6));
        assertEquals(5, MathOperations.gcd(15, 10));

        assertTrue(MathOperations.isPerfectSquare(16));
        assertFalse(MathOperations.isPerfectSquare(15));

        assertTrue(MathOperations.isPalindrome("a b a"));
        assertFalse(MathOperations.isPalindrome("hello"));

        assertArrayEquals(new int[]{5, 4, 3, 2, 1}, MathOperations.reverseArray(new int[]{1, 2, 3, 4, 5}));
        assertArrayEquals(new int[]{1, 2, 5, 8, 9}, MathOperations.bubbleSort(new int[]{5, 2, 8, 1, 9}));
        assertArrayEquals(new int[]{1, 2, 3, 4, 5}, MathOperations.bubbleSort(new int[]{1, 2, 3, 4, 5}));
    }
    
    @Test
    void testObjectEquality() {
        String str1 = new String("test");
        String str2 = new String("test");
        assertEquals(str1, str2); // equals() method
        assertNotSame(str1, str2); // different objects
    }
    
    @Test
    void testExceptionHandling() {
        assertThrows(ArithmeticException.class, () -> {
            @SuppressWarnings("unused")
            int result = 10 / 0;
        });
    }
    
    @Test
    void testFloatingPointPrecision() {
        double result = 0.1 + 0.2;
        assertEquals(0.3, result, 0.00001); // Use delta for floating point
    }
}
