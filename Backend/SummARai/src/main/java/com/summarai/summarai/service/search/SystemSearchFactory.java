package com.summarai.summarai.service.search;
import com.summarai.summarai.dto.BookSearchRequest;
import com.summarai.summarai.mapper.BookSummaryMapper;
import com.summarai.summarai.model.BookSummary;
import com.summarai.summarai.repository.BookSummaryRepository;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.domain.Specification;
import org.springframework.stereotype.Service;

@Service
public class SystemSearchFactory extends SearchFactory{
    private final BookSummaryRepository bookSummaryRepository;
    private final BookSummaryMapper mapper;

    public SystemSearchFactory(BookSummaryRepository bookSummaryRepository, BookSummaryMapper mapper) {
        this.bookSummaryRepository = bookSummaryRepository;
        this.mapper = mapper;
    }
    @Override
    public Page<?> CreateSearch(BookSearchRequest bookSearchRequest, Pageable pageable) {
        this.criteria = new SystemSearchCriteria();
        Specification<?> spec =criteria.search(bookSearchRequest);
        return bookSummaryRepository.findAll((Specification<BookSummary>) spec,pageable).map(mapper::toDto);
    }
}
