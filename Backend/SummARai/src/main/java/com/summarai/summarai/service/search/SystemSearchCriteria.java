package com.summarai.summarai.service.search;

import com.summarai.summarai.dto.BookSearchRequest;
import com.summarai.summarai.dto.BookSpecs;
import com.summarai.summarai.dto.SummaryDto;
import com.summarai.summarai.model.BookSummary;
import com.summarai.summarai.model.Genre;
import com.summarai.summarai.model.Summary;
import com.summarai.summarai.repository.BookSummaryRepository;
import com.summarai.summarai.service.impl.Normalizer;
import jakarta.persistence.criteria.Join;
import jakarta.persistence.criteria.JoinType;
import org.springframework.data.domain.Page;
import org.springframework.data.jpa.domain.Specification;

import java.util.List;

public class SystemSearchCriteria extends SearchCriteria {
    private String author;
    private List<String> genres;

    public String getAuthor() { return author; }
    public void setAuthor(String author) { this.author = author; }

    public List<String> getGenres() { return genres; }
    public void setGenres(List<String> genres) { this.genres = genres; }

    @Override
    Specification<BookSummary> search(BookSearchRequest searchRequest) {
        Specification<BookSummary> spec = Specification.where(null);
        searchRequest.setAuthor(Normalizer.normalizeArabic(searchRequest.getAuthor()));
        searchRequest.setTitle(Normalizer.normalizeArabic(searchRequest.getTitle()));

        if (searchRequest.getAuthor() != null && !searchRequest.getAuthor().isEmpty()) {
            spec = spec.and(authorContains(searchRequest.getAuthor()));
        }
        if (searchRequest.getTitle() != null && !searchRequest.getTitle().isEmpty()) {
            String normTitle = Normalizer.normalizeArabic(searchRequest.getTitle());
            spec = spec.and(normTitleContains(normTitle));
        }
        List<String> genres = searchRequest.getGenres();
        if (genres != null && !genres.isEmpty()) {
            spec = spec.and(hasAnyGenre(genres));
        }

        return spec;
    }
    public static Specification<BookSummary> hasAnyGenre(List<String> genres) {
        return (root, query, cb) -> {
            query.distinct(true);
            Join<BookSummary, Genre> join = root.join("genres", JoinType.INNER);
            return join.get("description").in(genres);
        };
    }
    public static Specification<BookSummary> authorContains(String author) {
        return (root, query, cb) ->
                cb.like(root.get("author").get("name"), "%" + author + "%");
    }
    public static Specification<BookSummary> normTitleContains(String norm_title) {
        return (root, query, cb) ->
                cb.like(cb.lower(root.get("normTitle")), "%" + norm_title+ "%");
    }


}
