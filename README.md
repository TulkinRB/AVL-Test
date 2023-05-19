# AVL-Test
A statistical tester for the AVL Tree exercise of the course "Data Structures".

## Setup
The tester uses the package `tqdm` to create a fancy progress bar, so you need to have it installed. Install it with
```commandline
pip install tqdm
```

## Usage
- Copy the code file `tester.py` to the same directory as your avl code.
- The code assumes that your file is named `AVLTree.py` and that the tree class is named `AVLTree`
- The code also assumes that your node class has the method `get_balance_factor()` (not used on virtual nodes).
  - If you don't have it the tester will run, but won't validate the balance factors of the nodes - it is recommended to create one.
- For each iteration of the test, a random test-case (sequence of operations) is generated and performed.
  - If a test fails, an exception will be raised. The exception contains the steps used to produce the error. Above this exception
    (in the error output) you can see the actual exception raised - either by the avl tree code or by the tester's validators.
  - The next time the tester runs after a failure it will automatically pick the same test-case - so you can debug your code using the same inputs.
- To run - simply execute `tester.py`

### Bulk testing
- In bulk mode, failed tests will not stop the tester. Instead, the failures will be saved and reported after the run.
- The next time the tester runs after a bulk run with failures, will perform all failed test-cases, if it is still being run in bulk mode with failures, will perform all failed test-cases, if it is still being run in bulk mode.
- This mode is intended for longer runs, after shorter runs seem to work. Exception tracebacks are not printed in this mode, so
  it is less ideal if bugs are to be expected.
- To enable - set `BULK_MODE` to `True` in the tester code.


## Configuration

There are a few configurations you can edit for the tester, they appear as constants in the tester code.

You don't need to edit those, but you can if you want :)
- `BULK_MODE` - A flag to turn on bulk mode
- `NUM_OF_TESTS` - The number of test iterations the tester performs - higher is better for test coverage but takes longer to run.
- `NUM_OF_STEPS` - The number of operations each test iterations performs - higher is better for test coverage but takes longer to run.
  - Somewhere between 256 and 1024 seems to be the maximum practical value for large iteration counts - depending on your machine.
- `MIN_KEY` and `MAX_KEY` - The range of the generated keys. It's recommended to choose a range with a size of a 1 order of magnitude above the number of steps. Don't make it too large since it seems to slow the tester down for some reason...
- `STEP_WEIGHTS` - Each step has a minimum weight and maximum weight - a random value in selected each iteration. The weight is the relative number of times this step will be executed.
- `RESULT_FILE_PATH` - This is the path to the file that the tester uses to save the steps if a test fails.
  This is used by the tester to re-run failed tests with the same test case.
  By default, the path is "<home-directory>/avl_tester_results.json".
