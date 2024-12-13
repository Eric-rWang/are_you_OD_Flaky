from Levenshtein import distance

def calculate_edit_distances(predictions, references):

    return [distance(pred, ref) for pred, ref in zip(predictions, references)]

predictions = ["def test_cached(cache_obj):\n    # Fix: Use separate cache objects for each test function\n    cache_obj1 = cache_obj.create_new_instance()\n    cache_obj2 = cache_obj.create_new_instance()\n\n    for _ in range(10):\n        cache_obj1.test1(8, 0)\n\n    # Fix: Use the correct cache object for the assertion\n    assert len(cache_obj1) == 1\n    assert cache_obj1.test1(8, 0) == 1\n\n    for _ in range(10):\n        cache_obj2.test2()\n\n    # Fix: Use the correct cache object for the assertion\n    assert cache_obj2.test2() == 1\n    assert len(cache_obj2) == 1\n\n    # Fix: Clear the correct cache object\n    cache_obj2.clear()\n    assert len(cache_obj2) == 0\n\n    # Make sure the cached function is properly wrapped\n    assert cache_obj2.test2.__doc__ == 'running test2'"]
references = ["def test_cached(cache_obj):\n    for _ in range(10):\n        cache_obj.test3(8, 0)\n        cache_obj.test4()\n    assert (len(c) == 1)\n    key = list(c.keys())[0]\n    assert (key == 'asdf')\n    c.clear()\n\n    for _ in range(10):\n        cache_obj.test1(8, 0)\n    assert len(c) == 1\n    assert cache_obj.test1(8, 0) == 1\n\n    for _ in range(10):\n        cache_obj.test2()\n    assert cache_obj.test2() == 1\n    assert len(c) == 2\n\n    c.clear()\n    assert len(c) == 0\n\n    # Make sure the cached function is properly wrapped\n    assert cache_obj.test2.__doc__ == 'running test2'"]
distances = calculate_edit_distances(predictions, references)
print(distances)
