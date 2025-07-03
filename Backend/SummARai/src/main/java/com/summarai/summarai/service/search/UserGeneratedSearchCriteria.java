package com.summarai.summarai.service.search;

import com.summarai.summarai.dto.BookSearchRequest;
import com.summarai.summarai.dto.BookSpecs;
import com.summarai.summarai.model.UserSummary;
import com.summarai.summarai.service.impl.Normalizer;
import org.springframework.data.jpa.domain.Specification;

public class UserGeneratedSearchCriteria extends SearchCriteria{
    @Override
    Specification<UserSummary> search(BookSearchRequest bookSearchRequest) {
        Specification<UserSummary> spec = Specification.where(null);
        bookSearchRequest.setTitle(Normalizer.normalizeArabic(bookSearchRequest.getTitle()));

        if (bookSearchRequest.getTitle() != null && !bookSearchRequest.getTitle().isEmpty()) {
            String normTitle = Normalizer.normalizeArabic(bookSearchRequest.getTitle());
            spec = spec.and(normTitleContains(normTitle));
        }
        spec = spec.and(isPublic());
        return spec;
    }
    public static Specification<UserSummary> normTitleContains(String norm_title) {
        return (root, query, cb) ->
                cb.like(cb.lower(root.get("normTitle")), "%" + norm_title+ "%");
    }
    public static Specification<UserSummary> isPublic() {
        return (root, query, cb) ->
                cb.equal(root.get("is_public"),true);
    }
}
