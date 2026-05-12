package dataloader;

import java.util.Arrays;

/**
 * Utility class for mathematical and algorithmic operations.
 * Used as target class for EvoSuite automated test generation.
 */
public class MathOperations {
    
    private MathOperations() {
        // Private constructor to hide the implicit public one
    }
    
    /**
     * Calculate the sum of an array of integers.
     */
    public static int sum(int[] numbers) {
        if (numbers == null || numbers.length == 0) {
            return 0;
        }
        int total = 0;
        for (int num : numbers) {
            total += num;
        }
        return total;
    }
    
    /**
     * Find the maximum value in an array.
     */
    public static int max(int[] numbers) {
        if (numbers == null || numbers.length == 0) {
            throw new IllegalArgumentException("Array cannot be null or empty");
        }
        return Arrays.stream(numbers)
                .max()
                .orElseThrow(() -> new IllegalArgumentException("Array cannot be null or empty"));
    }
    
    /**
     * Find the minimum value in an array.
     */
    public static int min(int[] numbers) {
        if (numbers == null || numbers.length == 0) {
            throw new IllegalArgumentException("Array cannot be null or empty");
        }
        return Arrays.stream(numbers)
                .min()
                .orElseThrow(() -> new IllegalArgumentException("Array cannot be null or empty"));
    }
    
    /**
     * Check if a number is prime.
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
        for (int i = 3; i * i <= num; i += 2) {
            if (num % i == 0) {
                return false;
            }
        }
        return true;
    }
    
    /**
     * Calculate factorial of a number.
     */
    public static int factorial(int n) {
        if (n < 0) {
            throw new IllegalArgumentException("Factorial not defined for negative numbers");
        }
        if (n == 0 || n == 1) {
            return 1;
        }
        int result = 1;
        for (int i = 2; i <= n; i++) {
            result *= i;
        }
        return result;
    }
    
    /**
     * Check if a string is a palindrome.
     */
    public static boolean isPalindrome(String str) {
        if (str == null || str.isEmpty()) {
            return true;
        }
        String clean = str.toLowerCase().replaceAll("[^a-z0-9]", "");
        return clean.contentEquals(new StringBuilder(clean).reverse());
    }
    
    /**
     * Calculate Fibonacci number at position n.
     */
    public static int fibonacci(int n) {
        if (n < 0) {
            throw new IllegalArgumentException("Fibonacci not defined for negative numbers");
        }
        if (n == 0) {
            return 0;
        }
        if (n == 1) {
            return 1;
        }
        int a = 0;
        int b = 1;
        for (int i = 2; i <= n; i++) {
            int temp = a + b;
            a = b;
            b = temp;
        }
        return b;
    }
    
    /**
     * Find GCD (Greatest Common Divisor) of two numbers.
     */
    public static int gcd(int a, int b) {
        if (b == 0) {
            return a;
        }
        return gcd(b, a % b);
    }
    
    /**
     * Check if a number is a perfect square.
     */
    public static boolean isPerfectSquare(int num) {
        if (num < 0) {
            return false;
        }
        int sqrt = (int) Math.sqrt(num);
        return sqrt * sqrt == num;
    }
    
    /**
     * Reverse an array.
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
     * Sort array in ascending order (bubble sort).
     */
    public static int[] bubbleSort(int[] arr) {
        if (arr == null) {
            return null;
        }
        int[] sorted = arr.clone();
        Arrays.sort(sorted);
        return sorted;
    }
}
