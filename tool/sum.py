"""
Sinlge digit addition tool for nanoGPT experiment. 

The tool provides perfect sums for single digit numbers, 
which nanoGPT can call when doing addition tasks.
"""
def sum(num1, num2): 
    """ Add two single-digit numbers.
    
    This function is intentionally simple and perfect - it never makes
    mistakes. The challenge is teaching the GPT to use it correctly.
    
    Args:
        a (int): First digit, must be 0-9
        b (int): Second digit, must be 0-9
    
    Returns:
        int: Sum of a and b (can be 0-18)
    
    Raises:
        AssertionError: If inputs are not valid single digits
    
    Examples:
        >>> single_digit_add(3, 5)
        8
        >>> single_digit_add(9, 9)
        18
        >>> single_digit_add(0, 7)
        7"""
    assert isinstance(num1, int) and isinstance(num2, int), "Inputs must be integers"
    assert 0 <= num1 <= 9, "First number must be a single digit (0-9)"
    assert 0 <= num2 <= 9, "Second number must be a single digit (0-9)"
    result = num1 + num2

    return result

def testSum(): 
    """ Test the sum function with all combinations of single-digit inputs. """
    
    print("Test 1: Edge cases")
    assert sum(0, 0) == 0, "0+0 should equal 0"
    assert sum(9, 9) == 18, "9+9 should equal 18"
    assert sum(0, 5) == 5, "0+5 should equal 5"
    assert sum(7, 0) == 7, "7+0 should equal 7"
    print("  ✓ Edge cases work")
    
    print("Test 2: All combinations of single digits")
    for num1 in range(10):
        for num2 in range(10):
            expected = num1 + num2
            actual = sum(num1, num2)
            assert actual == expected, f"Test failed for {num1} + {num2}: expected {expected}, got {actual}"
    print("All tests passed!")

    print("Test 3: Error handling")
    try: 
        sum(-1, 5)
        assert False, "Should have raised an AssertionError"
    except AssertionError:
        print("  ✓ Correctly raised AssertionError for negative input")
    try: 
        sum("2", 5)
        assert False, "Should have raised error for string input"
    except AssertionError as e:
        print(f"  ✓ Correctly rejected string: {str(e)[:40]}...")


if __name__ == "__main__":  
    testSum()

    print("Example usage:")
    print(f"sum(3, 5) = {sum(3, 5)}")
    print(f"sum(9, 9) = {sum(9, 9)}")