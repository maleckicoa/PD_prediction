__What is the application about?__

PD_prediction is a Dockerized application (Docker Hub: maleckicoa/pd_pred) that returns the probability of default for a set of given loans.
The application is based on a real world dataset from one large European fintech company (see: dataset.csv).


__How to use the pd_pred application?__

- Install and run the Docker application.
- Open a terminal and run "docker run -ti maleckicoa/pd_pred" to load and run the image from Docker Hub.
- Open a new terminal, run "docker ps", then copy the container_ID of the running pd_pred container.
- Run "docker exec -it container_ID sh" to open the shell of the running pd_pred container.
- Paste the *curl POST request (which holds loan information) into the open container shell, and it will return the loan IDs together with their default probabilities

Note that the application supports:
  - curl requests which hold a dictionary with loan information (see example in curl_request.txt)
  - curl requests which reference a JSON file with loan information (see example in curl_file.txt, see examples of JSON files: loan1.json, loan5.json, loan20.json)


__How does the application work?__

Running the maleckicoa/pd_pred application triggers the main.py script which then: 1) starts a FastAPI application, 2) imports the trained model object, 3) imports the PD_model_train.py script.
FastAPI application receives loans information through curl POST requests which it then passes to the trained model object. 
The trained model object calculates the probabilites of default for the given loan information and returns back the default probabilites. 
The trained model object is a serialized (mod.pkl file) which can be re-trained by directly running the PD_model.train.py script as __main_ _.
The PD_model_train.py script is the core of the application, it holds 3 classes:
- TrainValTest class object splits the dataset into train/val/test subsets
- WoeEncode class object encodes the data subsets using Weight of Evidence encoding
- Model class object trains the XG-boost classifier and returns the probabilities od default


__Can I run the application without Docker?__

Yes, to run the application without Docker:
- setup a Python environment (preferably 3.11) and all the dependencies (requirements.txt).
- Clone this project repository localy and cd into it.
- Generate the trained model object (mod.pkl), by directly running the PD_model_train.py script as __main_ _ (this step is done only when running the application for the 1st time).
- Run the main.py script which will import the traned model object and start the application.
- Open a new terminal, cd into the project repository and paste a curl request, as explained before.

__Other information__

The repository also has a PD EDA & Model Run.ipynb jupyter notebook with the exploratory data analysis

If you wish make changes to the source code of the container:
- create a new local directory
- start an existing maleckicoa/pd_pred image container (or run a new container)
- copy the container files into the local folder "docker cp container_ID: /usr/src/app/. /my_local_directory"
- close the container
- run the maleckicoa/pd_pred_image (this will make a new container) and attach volumes to the local folder: "docker run -v /my_local_directory:/usr/src/app maleckicoa/pd_pred"
- you can now make changes to the source code in your local directory. The changes will be reflected in this particular container, even if you stop it and start it again.

See the Useful_Docker_Commands.txt file for easier interaction with Docker


