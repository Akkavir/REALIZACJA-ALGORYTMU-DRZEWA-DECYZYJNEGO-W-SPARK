import unittest
import indexBG

class TestMain(unittest.TestCase):

    def test_main_no_errors(self):
        for _ in range(10):
            try:
                indexBG.main()
            except Exception as e:
                raise AssertionError(f"main() znaleziono błędy: {e}")
            else:
                print("Nie znaleziono błędów")

if __name__ == "__main__":
    unittest.main()
