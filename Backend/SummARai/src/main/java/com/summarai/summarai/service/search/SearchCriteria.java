package com.summarai.summarai.service.search;
import com.summarai.summarai.dto.BookSearchRequest;
import org.springframework.data.jpa.domain.Specification;
import org.springframework.stereotype.Service;

import java.util.List;
@Service
public abstract class SearchCriteria {
    String title;
    abstract Specification<?> search(BookSearchRequest bookSearchRequest);
    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

}
