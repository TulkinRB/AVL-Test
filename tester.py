import json
import random
import bisect
from pathlib import Path

from tqdm import tqdm

from AVLTree import AVLTree


BULK_MODE = False

NUM_OF_TESTS = 1000
NUM_OF_STEPS = 512

MIN_KEY = 0
MAX_KEY = 10000

step_weights = {
    "insert": (8, 30),
    "delete": (8, 30),
    "split": (2, 8),
    "join": (3, 8),
}

RESULT_FILE_PATH = Path.home() / "avl_tester_results.json"


class TestFailedException(Exception):

    def __init__(self, steps=None, *args):
        super().__init__(*args)
        self.steps = steps

    def __str__(self):
        return super().__str__() + "\nPerformed Steps: " + str(self.steps)


def run():
    prev_steps = []
    try:
        with open(RESULT_FILE_PATH, "r") as result_file:
            raw_result = result_file.read()
            if raw_result != "":
                prev_results = json.loads(raw_result)
                prev_steps = [result[1] for result in prev_results]
    except FileNotFoundError:
        pass

    try:
        if len(prev_steps) > 0:
            exceptions = redo_tests(prev_steps)
        else:
            exceptions = do_new_tests()
        save_errors(exceptions)
    except TestFailedException as e:
        save_errors([(e, e.steps)])
        raise


def format_exception(exception):
    lines = list()
    while exception is not None:
        lines.append(repr(exception))
        exception = exception.__cause__
    return lines


def save_errors(errors):
    errors = [(format_exception(e), steps) for e, steps in errors]
    with open(RESULT_FILE_PATH, "w") as result_file:
        json.dump(errors, result_file)


def redo_tests(step_lists):
    exceptions = list()
    for steps in tqdm(step_lists):
        test = Test()
        try:
            test.redo_tests(steps)
        except TestFailedException as e:
            if not BULK_MODE:
                raise e
            exceptions.append((e.__cause__, e.steps))
        except Exception as e:
            if not BULK_MODE:
                raise e
            exceptions.append((e, []))
    print(f"{len(exceptions)} out of {len(step_lists)} tests failed.")
    return exceptions


def do_new_tests(num_of_tests=NUM_OF_TESTS):
    exceptions = list()
    for i in tqdm(range(num_of_tests)):
        test = Test()
        try:
            test.do_tests(NUM_OF_STEPS)
        except TestFailedException as e:
            if not BULK_MODE:
                raise e
            exceptions.append((e.__cause__, e.steps))
        except Exception as e:
            if not BULK_MODE:
                raise e
            exceptions.append((e, []))
    print(f"{len(exceptions)} out of {num_of_tests} tests failed.")
    return exceptions


class Test:

    def __init__(self):
        self.key_lists = [list()]
        self.trees = [AVLTree()]
        self.step_weights = self._generate_weights()

    def _generate_weights(self):
        return {
            "insert": random.randint(*step_weights["insert"]),
            "delete": random.randint(*step_weights["delete"]),
            "split": random.randint(*step_weights["split"]),
            "join": random.randint(*step_weights["join"]),
        }

    def do_tests(self, num_of_steps):
        steps = list()
        for i in range(num_of_steps):
            step = self._generate_step()
            steps.append(step)
            try:
                self._perform_step(step)
            except Exception as e:
                raise TestFailedException(steps) from e

    def redo_tests(self, steps):
        for step in steps:
            try:
                self._perform_step(step)
            except Exception as e:
                raise TestFailedException(steps) from e

    def _perform_step(self, step):
        if step[0] == "delete":
            self._perform_delete(step)
        if step[0] == "split":
            self._perform_split(step)
        if step[0] == "join":
            self._perform_join(step)
        if step[0] == "insert":
            self._perform_insert(step)
        self._check_state()

    def _perform_delete(self, step):
        step_type, tree, key = step
        self.key_lists[tree].remove(key)
        node = self.trees[tree].search(key)
        self.trees[tree].delete(node)

    def _perform_insert(self, step):
        step_type, tree, key = step
        bisect.insort(self.key_lists[tree], key)
        self.trees[tree].insert(key, (key, "value"))

    def _perform_split(self, step):
        step_type, tree, key = step
        key_index = self.key_lists[tree].index(key)
        left_list = self.key_lists[tree][:key_index]
        right_list = self.key_lists[tree][key_index + 1:]
        right_list.sort()
        node = self.trees[tree].search(key)
        left_tree, right_tree = self.trees[tree].split(node)
        self.trees[tree] = left_tree
        self.trees.insert(tree+1, right_tree)
        self.key_lists[tree] = left_list
        self.key_lists.insert(tree+1, right_list)

    def _perform_join(self, step):
        step_type, tree, key, is_right = step
        if is_right:
            llist = self.key_lists.pop(tree)
            ltree = self.trees.pop(tree)
            llist.append(key)
            self.key_lists[tree] = llist + self.key_lists[tree]
            self.key_lists[tree].sort()
            self.trees[tree].join(ltree, key, (key, "value"))
        else:
            rlist = self.key_lists.pop(tree + 1)
            rtree = self.trees.pop(tree + 1)
            self.key_lists[tree].append(key)
            self.key_lists[tree] += rlist
            self.key_lists[tree].sort()
            self.trees[tree].join(rtree, key, (key, "value"))

    def _generate_step(self):
        sizes = [len(lst) for lst in self.key_lists]
        possible_steps = list()
        possible_steps += ["insert"]*self.step_weights["insert"]
        if max(sizes) >= 1:
            possible_steps += ["delete"]*self.step_weights["delete"]
        if len(self.key_lists) > 1:
            possible_steps += ["join"]*self.step_weights["join"]
        if max(sizes) >= 1:
            possible_steps += ["split"]*self.step_weights["split"]

        step_type = random.choice(possible_steps)
        if step_type == "delete":
            return self._generate_delete()
        if step_type == "split":
            return self._generate_split()
        if step_type == "join":
            return self._generate_join()
        if step_type == "insert":
            return self._generate_insert()
        raise RuntimeError("Can't generate step :(")

    def _generate_delete(self):
        possible_trees = [index for index, lst in enumerate(self.key_lists) if len(lst) >= 1]
        tree = random.choice(possible_trees)
        key = random.choice(self.key_lists[tree])
        return "delete", tree, key

    def _generate_split(self):
        possible_trees = [index for index, lst in enumerate(self.key_lists) if len(lst) >= 1]
        tree = random.choice(possible_trees)
        key = random.choice(self.key_lists[tree])
        return "split", tree, key

    def _generate_join(self):
        for i in range(1000):
            left_tree = random.randint(0, len(self.key_lists) - 2)

            prev_tree_with_keys = left_tree
            diff = 0
            while True:
                if prev_tree_with_keys < 0:
                    min_key = MIN_KEY + diff
                    break
                if len(self.key_lists[prev_tree_with_keys]) > 0:
                    maxleft = self.key_lists[prev_tree_with_keys][-1]
                    min_key = maxleft + diff
                    break
                diff += 1
                prev_tree_with_keys -= 1

            next_tree_with_keys = left_tree + 1
            diff = 0
            while True:
                if next_tree_with_keys >= len(self.key_lists):
                    max_key = MAX_KEY - diff
                    break
                if len(self.key_lists[next_tree_with_keys]) > 0:
                    minright = self.key_lists[next_tree_with_keys][0]
                    max_key = minright - diff
                    break
                diff += 1
                next_tree_with_keys += 1

            if min_key+1 > max_key-1:
                continue
            key = random.randint(min_key + 1, max_key - 1)
            is_right = random.randint(0, 1) == 1
            return "join", left_tree, key, is_right
        raise RuntimeError("Can't generate a join step :(")

    def _generate_insert(self):
        for i in range(10000):
            tree = random.randint(0, len(self.key_lists) - 1)

            prev_tree_with_keys = tree - 1
            diff = 3
            while True:
                if prev_tree_with_keys < 0:
                    min_key = MIN_KEY + diff
                    break
                if len(self.key_lists[prev_tree_with_keys]) > 0:
                    maxleft = self.key_lists[prev_tree_with_keys][-1]
                    min_key = maxleft + diff
                    break
                diff += 3
                prev_tree_with_keys -= 1

            next_tree_with_keys = tree + 1
            diff = 3
            while True:
                if next_tree_with_keys >= len(self.key_lists):
                    max_key = MAX_KEY - diff
                    break
                if len(self.key_lists[next_tree_with_keys]) > 0:
                    minright = self.key_lists[next_tree_with_keys][0]
                    max_key = minright - diff
                    break
                diff += 3
                next_tree_with_keys += 1

            possible_keys = list(range(min_key, max_key + 1))
            for key in self.key_lists[tree]:
                try:
                    possible_keys.remove(key)
                except ValueError:
                    pass

            if len(possible_keys) > 0:
                key = random.choice(possible_keys)
                return "insert", tree, key
        raise RuntimeError("Can't generate an insert step :(")

    def _check_state(self):
        self._validate_trees()

        content_tests = {
            "rank": self._check_rank,
            "select": self._check_select,
            "in_order": self._check_inorder
        }
        for tree_index in range(len(self.key_lists)):
            exceptions = dict()
            for name, test in content_tests.items():
                try:
                    test(tree_index)
                except Exception as e:
                    exceptions[name] = e
            for name, e in exceptions.items():
                if not isinstance(e, AssertionError):
                    raise e
            if len(exceptions) == 3:
                assert False, "All three content checks on a tree failed. This is probably caused by a faulty tree-state"
            if len(exceptions) > 0:
                raise AssertionError(
                    f"The following content checks failed: {list(exceptions.keys())}. This is probably caused by "
                    f"faulty implementation of these methods."
                ) from next(iter(exceptions.values()))

    def _check_inorder(self, tree_index):
        tree = self.trees[tree_index]
        key_list = self.key_lists[tree_index]
        expected_inorder = ((key, (key, "value")) for key in key_list)
        received_inorder = tree.avl_to_array()
        assert len(received_inorder) == len(key_list), (
            f"In-order (avl_to_array) result length doesn't match expected value."
            f"Expected value: {len(key_list)}. Actual value: {len(received_inorder)}"
        )
        assert all(
            received_item == expected_item
            for received_item, expected_item
            in zip(received_inorder, expected_inorder)
        ), "In-order (avl_to_array) result doesn't match the expected result"

    def _check_rank(self, tree_index):
        tree = self.trees[tree_index]
        key_list = self.key_lists[tree_index]
        for i, key in enumerate(key_list):
            node = tree.search(key)
            assert node is not None, f"Unexpected result for search({key}): key not found"
            assert node.get_key() == key, f"Unexpected result for search({key}): returned node's key is {node.get_key()}."
            rank = tree.rank(node)
            assert rank == i + 1, (
                f"Unexpected result for rank({key}): {rank}. Expected result is {i + 1}"
            )

    def _check_select(self, tree_index):
        tree = self.trees[tree_index]
        key_list = self.key_lists[tree_index]
        for i, key in enumerate(key_list):
            selected_node = tree.select(i + 1)
            assert selected_node.get_key() == key, (
                f"Unexpected result for select({i + 1}): {selected_node.get_key()}. "
                f"Expected result is {key}"
            )

    def _validate_trees(self):
        for tree in self.trees:
            if tree.get_root() is not None:
                self._validate_node(tree.get_root())
                assert tree.root.get_parent() is None, "The root's parent is not None"

    def _validate_node(self, node):
        if not node.is_real_node():
            size = node.get_size()
            assert size == 0, f"Incorrect size of virtual node: {size}"
            height = node.get_height()
            assert height == -1, f"Incorrect height of virtual node: {height}"
            return -1, 0, None, None

        key = node.get_key()
        assert isinstance(key, int), f"Invalid key found: {key}. All keys used by the tester are integers"

        height_left, size_left, min_left_key, max_left_key = self._validate_node(node.get_left())
        height_right, size_right, min_right_key, max_right_key = self._validate_node(node.get_right())

        min_key = max_key = key

        if max_left_key is not None:
            assert max_left_key < key, f"Invalid BST: Node {max_left_key} is in the left subtree of node {key}"
            min_key = min_left_key
        if min_right_key is not None:
            assert min_right_key > key, f"Invalid BST: Node {min_right_key} is in the right subtree of node {key}"
            max_key = max_right_key

        tester_balance_factor = height_left - height_right
        assert abs(tester_balance_factor) < 2, f"Balance factor (computed by tester) of node {key} is {tester_balance_factor}"

        try:
            avl_balance_factor = node.get_balance_factor()
            assert avl_balance_factor == tester_balance_factor, (
                f"Incorrect balance factor for node {key}: {avl_balance_factor}. "
                f"Correct balance factor is {tester_balance_factor}"
            )
        except (AttributeError, TypeError):
            pass

        tester_height = max(height_left, height_right) + 1
        avl_height = node.get_height()
        assert avl_height == tester_height, f"Incorrect height for node {key}: {avl_height}. Correct height is {tester_height}"

        tester_size = size_left + size_right + 1
        avl_size = node.get_size()
        assert tester_size == avl_size, f"Incorrect size for node {key}: {avl_size}. Correct size is {tester_size}"

        left_key = node.get_left().get_key()
        left_parent_key = getattr(node.get_left().get_parent(), "get_key", lambda: None)()
        assert node.get_left().get_parent() is node, (
            f"Left child ({left_key}) of node {key} has parent {left_parent_key}"
        )
        right_key = node.get_right().get_key()
        right_parent_key = getattr(node.get_right().get_parent(), "get_key", lambda: None)()
        assert node.get_right().get_parent() is node, (
            f"Right child ({right_key}) of node {key} has parent {right_parent_key}"
        )

        return tester_height, tester_size, min_key, max_key


if __name__ == '__main__':
    run()
