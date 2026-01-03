package dataloader;

/**
 * Utility class for mathematical and algorithmic operations.
 * Used as target class for EvoSuite automated test generation.
 */
public class MathOperations {
    
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
        int maximum = numbers[0];
        for (int num : numbers) {
            if (num > maximum) {
                maximum = num;
            }
        }
        return maximum;
    }
    
    /**
     * Find the minimum value in an array.
     */
    public static int min(int[] numbers) {
        if (numbers == null || numbers.length == 0) {
            throw new IllegalArgumentException("Array cannot be null or empty");
        }
        int minimum = numbers[0];
        for (int num : numbers) {
            if (num < minimum) {
                minimum = num;
            }
        }
        return minimum;
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
        int left = 0;
        int right = clean.length() - 1;
        while (left < right) {
            if (clean.charAt(left) != clean.charAt(right)) {
                return false;
            }
            left++;
            right--;
        }
        return true;
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
        int a = 0, b = 1;
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
        for (int i = 0; i < sorted.length - 1; i++) {
            for (int j = 0; j < sorted.length - 1 - i; j++) {
                if (sorted[j] > sorted[j + 1]) {
                    int temp = sorted[j];
                    sorted[j] = sorted[j + 1];
                    sorted[j + 1] = temp;
                }
            }
        }
        return sorted;
    }
}
