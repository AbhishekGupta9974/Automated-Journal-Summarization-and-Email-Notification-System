package net.engineeringdigest.journalApp.service;

import net.engineeringdigest.journalApp.entity.JournalEntry;
import net.engineeringdigest.journalApp.entity.User;
import net.engineeringdigest.journalApp.repository.UserRepositay;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Service
public class JournalService {

    @Autowired
    private UserRepositay userRepositay;

    public String getLast7DaysJournal(String userName) {
        User user = userRepositay.findByUserName(userName);
        if (user == null) {
            return null;
        }

        LocalDateTime sevenDaysAgo = LocalDateTime.now(ZoneId.of("UTC")).minusDays(7);

        List<JournalEntry> recentEntries = user.getJournalEntries().stream()
                .filter(entry -> entry.getDate().isAfter(sevenDaysAgo))
                .collect(Collectors.toList());

        if (recentEntries.isEmpty()) {
            return "";
        }

        return recentEntries.stream()
                .map(JournalEntry::getContent)
                .collect(Collectors.joining(" "));
    }
}

