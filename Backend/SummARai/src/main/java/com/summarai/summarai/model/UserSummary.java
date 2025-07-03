package com.summarai.summarai.model;

import jakarta.persistence.*;

import javax.xml.crypto.Data;
import java.sql.Date;
import java.util.HashSet;
import java.util.Set;

@Entity
@DiscriminatorValue("USER")
public class UserSummary extends Summary {
    @Column
    private boolean is_public;
    @Column
    private  Date creation_time;

    public boolean isIs_public() {
        return is_public;
    }

    public void setIs_public(boolean is_public) {
        this.is_public = is_public;
    }

    public Date Creation_time() {
        return creation_time;
    }

    public void setCreation_time(Date creation_time) {
        this.creation_time = creation_time;
    }
}
