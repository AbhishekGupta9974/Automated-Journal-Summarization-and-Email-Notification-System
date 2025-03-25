package net.engineeringdigest.journalApp.service;

import net.engineeringdigest.journalApp.repository.UserRepositay;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.core.userdetails.User;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.stereotype.Component;
import org.springframework.stereotype.Service;

@Service
public class UserDetailServiceImpl implements UserDetailsService {

    @Autowired
    private UserRepositay userRepositay;

    @Override
    public UserDetails loadUserByUsername(String username) throws UsernameNotFoundException{

        net.engineeringdigest.journalApp.entity.User  user  = userRepositay.findByUserName(username);
        if(user != null){
            return User.builder()
                    .username(user.getUserName())
                    .password(user.getPassword())
                    .roles(user.getRoles().toArray(new String[0]))
                    .build();
        }

        throw new UsernameNotFoundException("user not found with username: " + username);
    }
}
