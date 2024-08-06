import unittest
from unittest import main
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', "..")))

from rag.util import FilterUtil

class TestFilterUtil(unittest.TestCase):    
    def test_direct_eq(self):
        filter = FilterUtil.from_dict({"genre": "comedy"})
        filter_dict = filter.dict()
        
        self.assertEqual(
            filter_dict, 
            {"$eq": {"key": "genre", "value": "comedy"}}
        )
    
    def test_nested_in(self):
        filter = FilterUtil.from_dict({"genre": {"$in": ["comedy", "action"]}})
        filter_dict = filter.dict()
        
        self.assertEqual(
            filter_dict, 
            {
                "$in": {"key": "genre", "value": ["comedy", "action"]}
            }
        )
    
    def test_logical(self):
        filter = FilterUtil.from_dict({"$and": [{"genre": "comedy"}, {"year": 2020}]})
        filter_dict = filter.dict()
        
        self.assertEqual(
            filter_dict, 
            {
                "$and": [
                    {"$eq": {"key": "genre", "value": "comedy"}},
                    {"$eq": {"key": "year", "value": 2020}}
                ]
            }
        )
        
        filter = FilterUtil.from_dict({"$and": [{"genre": {"$eq": "comedy"}}, {"year": {"$eq": 2020}}]})
        filter_dict = filter.dict()
        
        self.assertEqual(
            filter_dict, 
            {
                "$and": [
                    {"$eq": {"key": "genre", "value": "comedy"}},
                    {"$eq": {"key": "year", "value": 2020}}
                ]
            }
        )
        
        filter = FilterUtil.from_dict({"genre": "comedy", "year": 2020})
        filter_dict = filter.dict()
        
        self.assertEqual(
            filter_dict, 
            {
                "$and": [
                    {"$eq": {"key": "genre", "value": "comedy"}},
                    {"$eq": {"key": "year", "value": 2020}}
                ]
            }
        )
    
    def test_unmatchable_but_valid(self):
        filter = FilterUtil.from_dict({"$and": [{"genre": "comedy"}, {"genre": "action"}]})
        filter_dict = filter.dict()
        
        self.assertEqual(
            filter_dict, 
            {
                "$and": [
                    {"$eq": {"key": "genre", "value": "comedy"}},
                    {"$eq": {"key": "genre", "value": "action"}}
                ]
            }
        )

    def test_invalid_in(self):
        with self.assertRaises(ValueError):
            FilterUtil.from_dict({"genre": ["comedy", "action"]})
            
    def test_invalid_eq(self):
        with self.assertRaises(ValueError):
            FilterUtil.from_dict({"genre": {"$eq": ["comedy", "action"]}})
            
if __name__ == "__main__":
    main()