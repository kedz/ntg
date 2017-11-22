import unittest
import nt

class TestFieldReaderBase(unittest.TestCase):
    '''
    Test nt.dataio.field_reader.field_reader_base.FieldReaderBase class.
    '''
    
    def test_read_and_save(self):

        list_data = [1986, 3.14, 2000]
 
        class Dummy1(nt.dataio.field_reader.FieldReaderBase):
            def __init__(self, field):
                super(Dummy1, self).__init__(field)
                self.register_data("mydata")

            def read_extract(self, data):
                self.mydata.append(data)

            def finalize_saved_data(self):
                return tuple(self.mydata)
        instance1 = Dummy1(0)
        instance1.read(list_data)
        result = instance1.finalize_saved_data()
        self.assertTrue(len(result) == 1)
        self.assertTrue(result[0] == 1986)
        instance2 = Dummy1(2)
        instance2.read(list_data)
        result2 = instance2.finalize_saved_data()
        self.assertTrue(len(result2) == 1)
        self.assertTrue(result2[0] == 2000)

        dict_data = {"pi": 3.14, "not_pi": 1234}
        
        instance3 = Dummy1("pi")

        instance3.update_field_map(1)
        instance3.read(list_data)
        result3 = instance3.finalize_saved_data()
        self.assertTrue(len(result3) == 1)
        self.assertTrue(result3[0] == 3.14)
        instance3.reset_saved_data()

        instance3.update_field_map(1)
        instance3.read(list_data)
        result4 = instance3.finalize_saved_data()
        self.assertTrue(len(result4) == 1)
        self.assertTrue(result4[0] == 3.14)

    def test_register_data(self):
        '''
        Test register_data and reset_saved_data methods.
        '''

        class Dummy1(nt.dataio.field_reader.FieldReaderBase):
            def __init__(self, field):
                super(Dummy1, self).__init__(field)

            def read_extract(self):
                pass

            def finalize_saved_data(self):
                pass
        
        instance1 = Dummy1(0)
        self.assertFalse(hasattr(instance1, "mydata"))
        instance1.register_data("mydata")
        self.assertTrue(hasattr(instance1, "mydata"))
        instance1.mydata.append(3.14)
        self.assertTrue(instance1.mydata[0] == 3.14)
        instance1.reset_saved_data()
        self.assertTrue(len(instance1.mydata) == 0)

    def test_field(self):
        '''
        Test field property and field type.
        '''

        class Dummy1(nt.dataio.field_reader.FieldReaderBase):
            def __init__(self, field):
                super(Dummy1, self).__init__(field)

            def read_extract(self):
                pass

            def finalize_saved_data(self):
                pass
        
        instance1 = Dummy1("string_field")

        self.assertTrue(instance1.field == "string_field")
        self.assertTrue(instance1.field_type == str)

        instance2 = Dummy1(3)
        self.assertTrue(instance2.field == 3)
        self.assertTrue(instance2.field_type == int)

        with self.assertRaises(Exception) as context:
            broken_instance1 = Dummy1(3.14)
        
        self.assertEqual(
            "field must be of type str or int.",
            str(context.exception)) 

    def test_abstractness(self):
        '''
        Test exceptions are thrown for concrete instantions of FieldReaderBase
        that do not implement all abstract methods.
        '''

        # Case 1 user does not implement read_extract and finalize_saved_data
        # methods.
        class BrokenClass1(nt.dataio.field_reader.FieldReaderBase):
            def __init__(self):
                super(BrokenClass1, self).__init__(0)

        
        with self.assertRaises(TypeError) as context:
            broken_instance1 = BrokenClass1()

        self.assertEqual(
            "Can't instantiate abstract class BrokenClass1 with abstract" \
            " methods finalize_saved_data, read_extract",
            str(context.exception))

        # Case 2 user does not implement read_extract method.
        class BrokenClass2(nt.dataio.field_reader.FieldReaderBase):
            def __init__(self):
                super(BrokenClass2, self).__init__(0)

            def finalize_saved_data(self):
                pass
        
        with self.assertRaises(TypeError) as context:
            broken_instance2 = BrokenClass2()
            
        self.assertEqual(
            "Can't instantiate abstract class BrokenClass2 with abstract" \
            " methods read_extract",
            str(context.exception))

        # Case 3 user does not implement finalize_saved_data method.
        class BrokenClass3(nt.dataio.field_reader.FieldReaderBase):
            def __init__(self):
                super(BrokenClass3, self).__init__(0)

            def read_extract(self):
                pass

        with self.assertRaises(TypeError) as context:
            broken_instance3 = BrokenClass3()
            
        self.assertEqual(
            "Can't instantiate abstract class BrokenClass3 with abstract" \
            " methods finalize_saved_data",
            str(context.exception))

if __name__ == '__main__':
    unittest.main()
