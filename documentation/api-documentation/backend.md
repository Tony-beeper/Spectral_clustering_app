# Backend Flask Endpoints

### Homepage

- URL: `GET /`

  - content-type: `application/json`

- URL: `POST /index`(form submission)
  - content-type: `application/json`
  - body: object
    - sample_size: (number) number of points to have on the graph
    - clusters: (number) number of clusters to partition the graph into
    - noise: (number) graphical deviation of the points

### Step-by-Step algorithm

- URL: `GET /step1`
  - content-type: `application/json`
- URL: `GET /step2`
  - content-type: `application/json`
- URL: `GET /step3`
  - content-type: `application/json`
- URL: `GET /step4`
  - content-type: `application/json`
- URL: `GET /step5`
  - content-type: `application/json`
- URL: `GET /step6`
  - content-type: `application/json`

### Algorithm all-in-one

- URL: `GET /all`
  - content-type: `application/json`

### Credits

- URL: `GET /all`
  - content-type: `application/json`
