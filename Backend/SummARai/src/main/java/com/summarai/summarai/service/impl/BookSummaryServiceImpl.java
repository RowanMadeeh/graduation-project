package com.summarai.summarai.service.impl;

import com.summarai.summarai.dto.*;
import com.summarai.summarai.mapper.BookSummaryMapper;
import com.summarai.summarai.mapper.UserSummaryMapper;
import com.summarai.summarai.model.BookSummary;
import com.summarai.summarai.repository.BookSummaryRepository;
import com.summarai.summarai.repository.UserSummaryRepository;
import com.summarai.summarai.service.BookSummaryService;
import com.summarai.summarai.service.S3Service;
import com.summarai.summarai.service.search.SearchFactory;
import com.summarai.summarai.service.search.SystemSearchFactory;
import com.summarai.summarai.service.search.UserGeneratedSearchFactory;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.List;
import java.util.Optional;

@Service
public class BookSummaryServiceImpl implements BookSummaryService {
    private final BookSummaryRepository bookSummaryRepository;
    private final UserSummaryRepository userSummaryRepository;
    private final BookSummaryMapper bookSummaryMapper;
    private final UserSummaryMapper userSummaryMapper;
    private final S3Service s3Service;

    public BookSummaryServiceImpl(BookSummaryRepository summaryRepository, UserSummaryRepository userSummaryRepository, BookSummaryMapper summaryMapper, UserSummaryMapper userSummaryMapper, S3Service s3Service) {
        this.bookSummaryRepository = summaryRepository;
        this.userSummaryRepository = userSummaryRepository;
        this.bookSummaryMapper = summaryMapper;
        this.userSummaryMapper = userSummaryMapper;
        this.s3Service = s3Service;
    }
    public BookSummary saveBook(BookSummaryDto bookSummary, MultipartFile file) throws IOException {
        String filename =s3Service.uploadFile(file);
        bookSummary.setSummary_url(filename);
        BookSummary book =  bookSummaryMapper.toEntity(bookSummary);
        book.setNormTitle(Normalizer.normalizeArabic(book.getTitle()));
        return this.bookSummaryRepository.save(book);
    }
    public List<BookSummary> saveBooks(List<BookSummaryDto> bookSummaries,List<MultipartFile> file){
        return this.bookSummaryRepository.saveAll(bookSummaryMapper.toEntities(bookSummaries));
    }
    public String getSummaryURL(Long id){
        return this.bookSummaryRepository.findUrlById(id);
    }

    public Page<BookSummaryDto> getAllBooks(Pageable pageable){
        Page<BookSummary> books = bookSummaryRepository.findAll(pageable);
        return books.map(bookSummaryMapper::toDto);
    }

    public Optional<BookSummaryDto> getBookById(Long id) {
        return bookSummaryRepository.findById(id)
                .map(book -> bookSummaryMapper.toDto(book));
    }

    @Override
    public Page<BookSummaryDto> getBooksByAuthor(String author, Pageable pageable) {
        Page<BookSummary> books = bookSummaryRepository.findByAuthor_Name(author,pageable);
        return books.map(bookSummaryMapper::toDto);
    }

    @Override
    public Page<BookSummaryDto> getBooksByTitle(String title, Pageable pageable) {

        Page<BookSummary> books = bookSummaryRepository.findByNormTitle(Normalizer.normalizeArabic(title),pageable);
        return books.map(bookSummaryMapper::toDto);
    }
    @Override
    public Page<?> searchBooks(BookSearchRequest criteria, Pageable pageable) {
        SearchFactory searchType;
        if(criteria.getType()== SummaryType.BOOK) {
            searchType = new SystemSearchFactory(bookSummaryRepository, bookSummaryMapper);
        }

        else
            searchType = new UserGeneratedSearchFactory(userSummaryRepository,userSummaryMapper);

        return searchType.CreateSearch(criteria,pageable);
    }

//    @Transactional
//    public void normalizeExistingBooks(){
//        List<BookSummary> books = bookSummaryRepository.findAll();
//        for(BookSummary book : books){
//            book.setNorm_title(normalizeArabic(book.getTitle()));
//            System.out.println(book.getNorm_title());
//        }
//        bookSummaryRepository.saveAll(books);
//    }




}
