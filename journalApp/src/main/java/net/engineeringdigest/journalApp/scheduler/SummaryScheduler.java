package net.engineeringdigest.journalApp.scheduler;

import net.engineeringdigest.journalApp.entity.User;
import net.engineeringdigest.journalApp.repository.UserRepositay;
import net.engineeringdigest.journalApp.service.EmailService;
import net.engineeringdigest.journalApp.service.JournalService;
import net.engineeringdigest.journalApp.service.SummarizationService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;
import org.springframework.stereotype.Service;
import java.util.List;
import java.util.Map;

@Component
public class SummaryScheduler {

    @Autowired
    private UserRepositay userRepositay;

    @Autowired
    private JournalService journalService;

    @Autowired
    private SummarizationService summarizationService;

    @Autowired
    private EmailService emailService;

//    @Scheduled(cron = "0 0 12 */7 * *") // Runs every 7 days at 12:00 PM
    @Scheduled(cron = "0 */1 * * * *")
    public void sendWeeklySummaries() {
        List<User> users = userRepositay.findAll();

        for (User user : users) {
            String journalText = journalService.getLast7DaysJournal(user.getUserName());
            if (journalText == null || journalText.isEmpty()) {
                continue;
            }

            // Call the Flask API
            Map<String, String> summaryData = summarizationService.getSummary(journalText);
            String summary = summaryData.get("summary");
            String accuracy = summaryData.get("accuracy");

            // Send email
            String emailContent = "Summary of your last 7 days' journal:\n\n" + summary + "\n\nAccuracy: " + accuracy;
            emailService.sendEmail(user.getEmail(), "Your Weekly Journal Summary", emailContent);
        }
    }
}

