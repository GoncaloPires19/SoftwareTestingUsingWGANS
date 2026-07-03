package dataloader;

import java.util.Arrays;

/**
 * Utility class for mathematical and algorithmic operations.
 * Used as the target class for EvoSuite automatic test generation
 * and PIT mutation testing in the WGAN-GP test quality evaluation pipeline.
 */
public class MathOperations {

    private MathOperations() {
        // Utility class — not instantiable
    }

    // -----------------------------------------------------------------------
    // Array aggregation
    // -----------------------------------------------------------------------

    /**
     * Returns the sum of all elements.
     *
     * @param numbers the input array; null or empty returns 0
     */
    public static int sum(int[] numbers) {
        if (numbers == null || numbers.length == 0) {
            return 0;
        }
        int total = 0;
        for (int n : numbers) {
            total += n;
        }
        return total;
    }

    /**
     * Returns the maximum element.
     *
     * @throws IllegalArgumentException if {@code numbers} is null or empty
     */
    public static int max(int[] numbers) {
        if (numbers == null || numbers.length == 0) {
            throw new IllegalArgumentException("Array cannot be null or empty");
        }
        int max = numbers[0];
        for (int i = 1; i < numbers.length; i++) {
            if (numbers[i] > max) {
                max = numbers[i];
            }
        }
        return max;
    }

    /**
     * Returns the minimum element.
     *
     * @throws IllegalArgumentException if {@code numbers} is null or empty
     */
    public static int min(int[] numbers) {
        if (numbers == null || numbers.length == 0) {
            throw new IllegalArgumentException("Array cannot be null or empty");
        }
        int min = numbers[0];
        for (int i = 1; i < numbers.length; i++) {
            if (numbers[i] < min) {
                min = numbers[i];
            }
        }
        return min;
    }

    /**
     * Returns the arithmetic mean of all elements.
     *
     * @throws IllegalArgumentException if {@code numbers} is null or empty
     */
    public static double average(int[] numbers) {
        if (numbers == null || numbers.length == 0) {
            throw new IllegalArgumentException("Array cannot be null or empty");
        }
        long total = 0;
        for (int n : numbers) {
            total += n;
        }
        return (double) total / numbers.length;
    }

    /**
     * Returns the median of the array (does not modify the original array).
     * For even-length arrays, returns the lower of the two middle values.
     *
     * @throws IllegalArgumentException if {@code numbers} is null or empty
     */
    public static double median(int[] numbers) {
        if (numbers == null || numbers.length == 0) {
            throw new IllegalArgumentException("Array cannot be null or empty");
        }
        int[] sorted = numbers.clone();
        Arrays.sort(sorted);
        int mid = sorted.length / 2;
        if (sorted.length % 2 == 0) {
            return (sorted[mid - 1] + sorted[mid]) / 2.0;
        }
        return sorted[mid];
    }

    /**
     * Counts how many times {@code target} appears in {@code numbers}.
     *
     * @param numbers the input array; null returns 0
     * @param target  the value to count
     */
    public static int countOccurrences(int[] numbers, int target) {
        if (numbers == null) {
            return 0;
        }
        int count = 0;
        for (int n : numbers) {
            if (n == target) {
                count++;
            }
        }
        return count;
    }

    /**
     * Clamps {@code value} to the range [{@code min}, {@code max}].
     *
     * @throws IllegalArgumentException if {@code min > max}
     */
    public static int clamp(int value, int min, int max) {
        if (min > max) {
            throw new IllegalArgumentException("min must be <= max");
        }
        if (value < min) return min;
        if (value > max) return max;
        return value;
    }

    // -----------------------------------------------------------------------
    // Number theory
    // -----------------------------------------------------------------------

    /**
     * Returns {@code true} if {@code num} is a prime number.
     * Negative numbers and 0, 1 are not prime.
     */
    public static boolean isPrime(int num) {
        if (num < 2) {
            return false;
        }
        if (num == 2) {
            return true;
        }
        if (num % 2 == 0) {
            return false;
        }
        for (int i = 3; (long) i * i <= num; i += 2) {
            if (num % i == 0) {
                return false;
            }
        }
        return true;
    }

    /**
     * Computes {@code n!} for {@code 0 <= n <= 12}.
     * Returns a {@code long} to avoid silent int overflow.
     *
     * @throws IllegalArgumentException if {@code n < 0}
     * @throws ArithmeticException      if {@code n > 20} (overflows long)
     */
    public static long factorial(int n) {
        if (n < 0) {
            throw new IllegalArgumentException("Factorial not defined for negative numbers");
        }
        if (n > 20) {
            throw new ArithmeticException("Factorial overflows long for n > 20");
        }
        long result = 1;
        for (int i = 2; i <= n; i++) {
            result *= i;
        }
        return result;
    }

    /**
     * Returns the {@code n}-th Fibonacci number (0-indexed).
     * Returns a {@code long} to safely handle up to {@code n = 92}.
     *
     * @throws IllegalArgumentException if {@code n < 0}
     * @throws ArithmeticException      if {@code n > 92} (overflows long)
     */
    public static long fibonacci(int n) {
        if (n < 0) {
            throw new IllegalArgumentException("Fibonacci not defined for negative numbers");
        }
        if (n > 92) {
            throw new ArithmeticException("Fibonacci overflows long for n > 92");
        }
        if (n == 0) return 0L;
        if (n == 1) return 1L;
        long a = 0, b = 1;
        for (int i = 2; i <= n; i++) {
            long temp = a + b;
            a = b;
            b = temp;
        }
        return b;
    }

    /**
     * Returns the Greatest Common Divisor of {@code a} and {@code b}.
     * Handles negative inputs correctly via {@code Math.abs}.
     * {@code gcd(0, 0)} returns 0.
     */
    public static int gcd(int a, int b) {
        a = Math.abs(a);
        b = Math.abs(b);
        while (b != 0) {
            int temp = b;
            b = a % b;
            a = temp;
        }
        return a;
    }

    /**
     * Returns {@code true} if {@code num} is a perfect square (>= 0).
     * Uses integer arithmetic to avoid floating-point precision errors.
     */
    public static boolean isPerfectSquare(int num) {
        if (num < 0) {
            return false;
        }
        int sqrt = (int) Math.sqrt(num);
        // Check sqrt and sqrt+1 to guard against floor-rounding precision errors
        return sqrt * sqrt == num || (sqrt + 1) * (sqrt + 1) == num;
    }

    // -----------------------------------------------------------------------
    // Array manipulation
    // -----------------------------------------------------------------------

    /**
     * Returns a new array with elements in reverse order.
     * Returns {@code null} if {@code arr} is null.
     */
    public static int[] reverseArray(int[] arr) {
        if (arr == null) {
            return null;
        }
        int[] reversed = new int[arr.length];
        for (int i = 0; i < arr.length; i++) {
            reversed[i] = arr[arr.length - 1 - i];
        }
        return reversed;
    }

    /**
     * Returns a new array with elements sorted in ascending order.
     * Returns {@code null} if {@code arr} is null.
     * The original array is never modified.
     */
    public static int[] bubbleSort(int[] arr) {
        if (arr == null) {
            return null;
        }
        int[] sorted = arr.clone();
        Arrays.sort(sorted);
        return sorted;
    }

    // -----------------------------------------------------------------------
    // String operations
    // -----------------------------------------------------------------------

    /**
     * Returns {@code true} if {@code str} reads the same forwards and backwards,
     * ignoring case and non-alphanumeric characters.
     * {@code null} and empty strings are considered palindromes.
     */
    public static boolean isPalindrome(String str) {
        if (str == null || str.isEmpty()) {
            return true;
        }
        String clean = str.toLowerCase().replaceAll("[^a-z0-9]", "");
        int left = 0, right = clean.length() - 1;
        while (left < right) {
            if (clean.charAt(left) != clean.charAt(right)) {
                return false;
            }
            left++;
            right--;
        }
        return true;
    }
}
