package main

import (
	"bytes"
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/joho/godotenv"
)

type AIModelConnector struct {
	Client *http.Client
}

type Inputs struct {
	Table map[string][]string `json:"table"`
	Query string              `json:"query"`
}

type Response struct {
	Answer      string   `json:"answer"`
	Coordinates [][]int  `json:"coordinates"`
	Cells       []string `json:"cells"`
	Aggregator  string   `json:"aggregator"`
}

func CsvToSlice(data string) (map[string][]string, error) {
	r := csv.NewReader(strings.NewReader(data))
	r.TrimLeadingSpace = true

	records, err := r.ReadAll()
	if err != nil {
		return nil, err
	}

	if len(records) < 2 {
		return nil, errors.New("CSV file must contain at least one row of data")
	}

	headers := records[0]
	result := make(map[string][]string)
	for _, header := range headers {
		result[header] = []string{}
	}

	for _, row := range records[1:] {
		for i, value := range row {
			result[headers[i]] = append(result[headers[i]], value)
		}
	}

	return result, nil
}

func (c *AIModelConnector) ConnectAIModel(payload interface{}, token string) (Response, error) {
	url := "https://api-inference.huggingface.co/models/google/tapas-base-finetuned-wtq"
	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return Response{}, err
	}

	req, err := http.NewRequest("POST", url, bytes.NewBuffer(payloadBytes))
	if err != nil {
		return Response{}, err
	}

	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", token))
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.Client.Do(req)
	if err != nil {
		return Response{}, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return Response{}, fmt.Errorf("failed to get valid response: %d %s", resp.StatusCode, resp.Status)
	}

	respBody, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return Response{}, err
	}

	var response Response
	err = json.NewDecoder(bytes.NewReader(respBody)).Decode(&response)
	if err != nil {
		return Response{}, err
	}

	return response, nil
}

func loadEnv() {
	err := godotenv.Load()
	if err != nil {
		log.Fatal("Error loading .env file")
	}
}

func main() {
	loadEnv()
	router := gin.Default()

	// Serve the HTML file at the root route
	router.LoadHTMLFiles("index.html")

	router.GET("/", func(c *gin.Context) {
		c.HTML(http.StatusOK, "index.html", nil)
	})

	router.POST("/ask", func(c *gin.Context) {
		// Load CSV data
		data, err := os.Open("data-series.csv")
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Error reading CSV file: %v", err)})
			return
		}
		defer data.Close()

		rowData, err := ioutil.ReadAll(data)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Error reading CSV file: %v", err)})
			return
		}

		// Convert CSV to slice
		table, err := CsvToSlice(string(rowData))
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Error converting CSV to slice: %v", err)})
			return
		}

		// Get query from request body
		var jsonData struct {
			Query string `json:"query"`
		}
		if err := c.BindJSON(&jsonData); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request"})
			return
		}

		// Prepare payload
		payload := Inputs{
			Table: table,
			Query: jsonData.Query,
		}

		// Initialize AI model connector
		client := &http.Client{}
		connector := &AIModelConnector{Client: client}

		// Connect to AI model
		token := os.Getenv("HUGGINGFACE_TOKEN")
		if token == "" {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "HUGGINGFACE_TOKEN is not set in the environment"})
			return
		}

		response, err := connector.ConnectAIModel(payload, token)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Error connecting to AI model: %v", err)})
			return
		}

		// Send response back to front-end
		c.JSON(http.StatusOK, response)
	})

	router.Run(":8080")
}
