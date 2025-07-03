package com.summarai.summarai.service.search;

import com.summarai.summarai.dto.BookSearchRequest;
import com.summarai.summarai.dto.SummaryDto;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.domain.Specification;

public abstract class SearchFactory {
    SearchCriteria criteria;
    public abstract Page<?> CreateSearch(BookSearchRequest bookSearchRequest, Pageable pageable);
    public SearchCriteria getTitle() {
        return criteria;
    }

    public void setTitle(SearchCriteria criteria) {
        this.criteria = criteria;
    }



}
