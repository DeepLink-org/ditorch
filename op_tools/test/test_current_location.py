# Copyright (c) 2024, DeepLink.
from op_tools.utils import current_location

import unittest


class TestCurrentLocation(unittest.TestCase):

    def test_current_location(self):
        def f1():
            def f2():
                def f3():
                    return current_location(stack_depth=-2, print_stack=True)

                return f3()

            return f2()

        location = f1()
        print(f"location: {location}")
        self.assertTrue(isinstance(location, str))
        self.assertTrue("f3" in location)
        self.assertTrue(__file__ in location)


if __name__ == "__main__":
    unittest.main()
