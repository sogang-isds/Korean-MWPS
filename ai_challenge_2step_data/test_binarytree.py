import unittest
from data_utils import convert_func_to_binary, convert_math_to_binary


class TestTokenizationMethods(unittest.TestCase):
    def test_convert_equation(self):
        string = 'func_1(func_2("A", null), num_0)'
        expected_string = 'func_1 func_2 "A" null num_0'

        actual_string = convert_func_to_binary(string)

        self.assertEqual(expected_string, actual_string)

    def test_convert_equation2(self):
        string = 'func_1(func_2("A", null), func_3("B", null))'
        expected_string = 'func_1 func_2 "A" null func_3 "B" null'

        actual_string = convert_func_to_binary(string)

        self.assertEqual(expected_string, actual_string)

    def test_convert_equation2_without_space(self):
        string = 'func_1(func_2("A",null),func_3("B",null))'
        expected_string = 'func_1 func_2 "A" null func_3 "B" null'

        actual_string = convert_func_to_binary(string)

        self.assertEqual(expected_string, actual_string)

    def test_convert_equation3(self):
        string = 'func_1("A", func_3("B", null))'
        expected_string = 'func_1 "A" func_3 "B" null'

        actual_string = convert_func_to_binary(string)

        self.assertEqual(expected_string, actual_string)

    def test_convert_equation4(self):
        # (5 + (6 / 3)) * ((6 / 3) - 7)
        string = 'multiply(add(5, divide(6, 3)), minus(divide(6, 3), 7))'
        expected_string = 'multiply add 5 divide 6 3 minus divide 6 3 7'

        actual_string = convert_func_to_binary(string)

        self.assertEqual(expected_string, actual_string)

    def test_convert_operation(self):
        string = '(5 + (6 / 3)) * ((6 / 3) - 7)'
        expected_string = '* + 5 / 6 3 - / 6 3 7'

        actual_string = convert_math_to_binary(string)

        self.assertEqual(expected_string, actual_string)

    def test_convert_operation2(self):
        string = '(5 + (6 / 3)) * ((6 / 3) - 7) / ((4 / 2) - 10)'
        expected_string = '/ * + 5 / 6 3 - / 6 3 7 - / 4 2 10'

        actual_string = convert_math_to_binary(string)

        self.assertEqual(expected_string, actual_string)

    def test_convert_operation3(self):
        string = '(5 + (6 / 3)) * 3 / ((4 / 2) - 10)'
        expected_string = '/ * + 5 / 6 3 3 - / 4 2 10'

        actual_string = convert_math_to_binary(string)

        self.assertEqual(expected_string, actual_string)

    def test_convert_operation4_without_space(self):
        string = '((((7*10)+6)-4)/3)'
        expected_string = '/ - + * 7 10 6 4 3'

        actual_string = convert_math_to_binary(string)

        self.assertEqual(expected_string, actual_string)


if __name__ == '__main__':
    unittest.main()