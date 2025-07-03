package com.summarai.summarai.dto;

import com.summarai.summarai.model.BookSummary;
import com.summarai.summarai.model.Genre;
import com.summarai.summarai.model.Summary;
import jakarta.persistence.criteria.Join;
import jakarta.persistence.criteria.JoinType;
import org.springframework.data.jpa.domain.Specification;

import java.util.List;

public class BookSpecs {

    public static Specification<Summary> authorContains(String author) {
        return (root, query, cb) ->
                cb.like(root.get("author").get("name"), "%" + author + "%");
    }

    public static Specification<Summary> normTitleContains(String norm_title) {
        return (root, query, cb) ->
                cb.like(cb.lower(root.get("normTitle")), "%" + norm_title+ "%");
    }

    /**
     * Matches books having any of the specified genres.
     */
    public static Specification<Summary> hasAnyGenre(List<String> genres) {
        return (root, query, cb) -> {
            // Avoid duplicate results
            query.distinct(true);
            Join<Summary, Genre> join = root.join("genres", JoinType.INNER);
            return join.get("description").in(genres);
        };
    }
}
