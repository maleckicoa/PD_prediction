What is the application about?

maleckicoa/pd_pred is a Dockerized application that returns the probability of default for a set of given loans
The application is based on a real dataset from one large European fintech company (see: dataset.csv).


How to use the pd_pred application?

- Install and run the Docker application
- navigate to the project folder and run "docker run -ti maleckicoa/pd_pred" - to load and run the image from Docker Hub
- open a new terminal, run "docker ps", then copy the <container ID> of the running pd_pred container
- run "docker exec -it <container ID> sh" to open the shell of the running pd_pred container
- paste the *curl POST request (which holds loan information) into the open container shell,
  and it will return the loan ID together with loan default probabilities

* Note that the application supports:
  -curl requests which hold a dictionary with loan information (see example in curl_request.txt)
  -curl requests which reference a JSON file with loan information (see request example in curl_file.txt,
  see examples of JSON files: loan1.json, loan5.json, loan20.json)


How does the application work?

The maleckicoa/pd_pred application consists of 2 python scripts.

The main.py script imports the PD_model_train.py script and starts a FastAPI application.
FastAPI application serves to receive loans information as curl POST requests and to return
the default probabilities it receives from the model object.

The PD_model_train.py holds 3 classes which respectively:
- split the dataset into train/val/test subsets (TrainValTest)
- encode the data subsets using Weight of Evidence encoding (WoeEncode)
- train the XG-boost classifier and return probability od default (Model)

Note that the model object in the image/container is pretrained and serialized (mod.pkl file).
At runtime the model object is imported and the FastAPI application passes the loan information
to the model object, which in return provides the default probabilities. The model object can be re-trained,
this is done by directly running the PD_model.train.py script as __main__.

Other Info:
To run the application locally (without Docker), one needs to setup a Python environment and all the dependencies
(requirements.txt). In addition, the model object (mod.pkl) needs to be generated, this is done by directly 
running the PD_model.train.py script as __main__. After that, running the main.py will start the application


The repository also has a PD EDA & Model Run.ipynb jupyter notebook with the exploratory data analysis

If the user wishes to make changes to the source code of the container, follow the steps:
- create a new local directory
- start an existing maleckicoa/pd_pred image container (or run a new container)
- copy the container files into the local folder "docker cp <container ID>: /usr/src/app/. /my_local_directory"
- close the container
- run the maleckicoa/pd_pred_image (this will make a new container) and attach volumes to the local folder
 "docker run -v /my_local_directory:/usr/src/app maleckicoa/pd_pred"
- you can now make changes to the source code in your local directory. The changes will be reflected in
  this particular container, even if you stop it and start it again.

See the Useful_Docker_Commands.txt file for easier interaction with Docker


