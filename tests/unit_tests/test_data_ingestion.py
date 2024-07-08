import os
import unittest


class TestIngestScript(unittest.TestCase):
    def setUp(self):
        # Set up the test environment
        self.data_dir = "./data/processed"
        self.dataset_file = "housing.csv"

    def test_dataset_download(self):
        # Run the ingest.py script
        os.system("python ingest.py")

        # Check if the dataset file is present
        dataset_path = os.path.join(self.data_dir, self.dataset_file)
        self.assertTrue(
            os.path.exists(dataset_path),
            f"Dataset file {self.dataset_file} not found in {self.data_dir}",
        )

        # Check the file size
        file_size = os.path.getsize(dataset_path)
        self.assertGreater(
            file_size, 0, f"Dataset file {self.dataset_file} is empty"
        )
        first_l = (
            "longitude,latitude,housing_median_age,total_rooms,total_bedrooms"
            + ",population,households,median_income,median_house_value,"
            + "ocean_proximity,income_cat"
        )
        # Check the file contents (example)
        with open(dataset_path, "r") as f:
            first_line = f.readline().strip()
            self.assertEqual(
                first_line,
                first_l,
                "First line of dataset does not match expected format",
            )


if __name__ == "__main__":
    unittest.main()
