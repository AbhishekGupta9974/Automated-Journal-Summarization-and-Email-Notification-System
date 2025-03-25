package net.engineeringdigest.journalApp.service;

import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.ResponseEntity;
import org.springframework.http.MediaType;
import java.util.HashMap;
import java.util.Map;

@Service
public class SummarizationService {

    private final String FLASK_API_URL = "http://localhost:5000/summarize"; // Flask API URL

    public Map<String, String> getSummary(String text) {
        RestTemplate restTemplate = new RestTemplate();
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);

        Map<String, String> requestBody = new HashMap<>();
        requestBody.put("text", text); // Only sending text

        HttpEntity<Map<String, String>> request = new HttpEntity<>(requestBody, headers);
        ResponseEntity<Map> response = restTemplate.postForEntity(FLASK_API_URL, request, Map.class);

        return response.getBody(); // Returns { "summary": "...", "accuracy": "..." }
    }
}

