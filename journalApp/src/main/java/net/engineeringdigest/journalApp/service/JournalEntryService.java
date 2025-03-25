package net.engineeringdigest.journalApp.service;

import lombok.extern.slf4j.Slf4j;
import net.engineeringdigest.journalApp.entity.JournalEntry;
import net.engineeringdigest.journalApp.entity.User;
import net.engineeringdigest.journalApp.repository.JournalEntryRepositay;
import org.bson.types.ObjectId;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

@Service
@Slf4j
public class JournalEntryService {
    @Autowired
    private JournalEntryRepositay journalEntryRepositay;

    @Autowired
    private UserService userService;



    @Transactional
    public void saveEntry(JournalEntry journalEntry, String userName){
        try {
            User user = userService.findByUsername(userName);
            journalEntry.setDate(LocalDateTime.now());
            JournalEntry saved = journalEntryRepositay.save(journalEntry);
            user.getJournalEntries().add(saved);
            userService.saveUser(user);
        }catch (Exception e){

            throw new RuntimeException("An error occured");
        }

    }

    public void saveEntry(JournalEntry journalEntry){
        journalEntryRepositay.save(journalEntry);
    }


    public List<JournalEntry> getAll(){
        return journalEntryRepositay.findAll();
    }

    public Optional<JournalEntry> findById(ObjectId Id) {
        return journalEntryRepositay.findById(Id);
    }

    @Transactional
    public boolean deleteById(ObjectId Id, String userName){
        boolean removed = false;
        try {
            User user = userService.findByUsername(userName);
            removed = user.getJournalEntries().removeIf(x -> x.getId().equals(Id));
            if(removed){
                userService.saveUser(user);
                journalEntryRepositay.deleteById(Id);
            }
        } catch (Exception e){
            System.out.println(e);
            throw new RuntimeException("An error occured while deleting the entry.",e);
        }
        return removed;
    }

//    public List<JournalEntry> findByUserName(String userName){
//
//    }
}
