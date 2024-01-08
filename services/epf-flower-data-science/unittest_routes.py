import os
import pytest
from fastapi.testclient import TestClient

class TestEndpointRoutes:
    @pytest.fixture
    def client(self) -> TestClient:
        """
        Test client for integration tests
        """

        from src.app import get_application

        app = get_application()

        client = TestClient(app, base_url="http://testserver")

        return client
    
    def test_get_data(self, client):
        """
        Test the /data endpoint of the API that dowload the data.
        
        Args:
            self: The current instance of the test class.
            client: The client object to use for making the GET request.
        
        Returns:
            None
        """
        response = client.get("/data")
        assert response.status_code == 200
        assert response.text.strip('"\n') == "ok"

    def test_load_data(self, client):
        """
        Test the /data/dowload endpoint of the API that load the data ad json.

        Returns:
            None
        """
        response = client.get("/data/dowload")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        assert response.headers["content-length"] == '17193'
        data = response.json()
        assert isinstance(data, list)
        for item in data:
            assert "Id" in item
            assert "SepalLengthCm" in item
            assert "SepalWidthCm" in item
            assert "PetalLengthCm" in item
            assert "PetalWidthCm" in item
            assert "Species" in item

    def test_process_data(self, client):
        """
        Test the /data/process endpoint of the API.

        Returns:
            None
        """

        response = client.get("/data/process")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        assert response.headers["content-length"] == '16443'
        data = response.json()
        assert isinstance(data, list)
        for item in data:
            assert "Id" in item
            assert "SepalLengthCm" in item
            assert "SepalWidthCm" in item
            assert "PetalLengthCm" in item
            assert "PetalWidthCm" in item
            assert "Species" in item
            assert not item['Species'].startswith('Iris-')

    def test_split_data(self, client):
        """
        Test the /data/split endpoint of the API that return a json with X_train, X_test, y_train, y_test as lists.

        Returns:
            None
        """
        response = client.get("/data/split")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        data = response.json()

        assert isinstance(data, list)
        assert len(data) == 4

        X_train, X_test, y_train, y_test = data
        assert isinstance(X_train, list)
        assert isinstance(X_test, list)
        assert isinstance(y_train, list)
        assert isinstance(y_test, list)

    def test_train_model(self, client):
        """
        Test the /data/train endpoint of the API that train and save the KNeigbors model.

        Returns:
            None
        """
        response = client.get("/data/train")
        assert response.status_code == 200
        assert response.text.strip('"\n') == "ok, model trained and saved"
        assert os.path.isfile('src/models/KNN_model.pkl')

    def test_predict_endpoint(self, client):
        """
        Test the predict endpoint of the API.

        Returns:
            None
        """
        SepalLengthCm = 5.1
        SepalWidthCm = 3.5
        PetalLengthCm = 1.4
        PetalWidthCm = 0.2
        response = client.get(f"/data/prediction?SepalLengthCm={SepalLengthCm}&SepalWidthCm={SepalWidthCm}&PetalLengthCm={PetalLengthCm}&PetalWidthCm={PetalWidthCm}")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        data = response.json()
        assert isinstance(data, list)

    def test_new_firestore_collection_parameters(self, client):
        """
        Test the new_firestore_collection_parameters endpoint with different new parameters.

        Returns:
            None
        """
        parameter_name = "test_param"
        parameter_value = 123
        response = client.get(f"/data/new_firestore_collection_parameters?parameter_name={parameter_name}&parameter_value={parameter_value}")
        data = response.json()

        assert response.status_code == 200
        assert parameter_name in data
        assert data[parameter_name] == parameter_value
