package com.summarai.summarai.service.impl;
import java.time.LocalDate;
import com.summarai.summarai.mapper.UserSummaryMapper;
import com.summarai.summarai.model.UserSummary;
import com.summarai.summarai.repository.UserSummaryRepository;
import com.summarai.summarai.security.UserDetailsServiceImpl;
import com.summarai.summarai.service.SummaraiService;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.core.io.Resource;
import org.springframework.http.MediaType;
import org.springframework.http.client.MultipartBodyBuilder;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;
import reactor.core.publisher.Mono;

import java.io.IOException;
import java.time.ZoneId;
import java.sql.Date;

class MultipartFileResource extends ByteArrayResource {

    private final String fileName;

    public MultipartFileResource(MultipartFile multipartFile) throws IOException {
        super(multipartFile.getBytes());
        this.fileName = multipartFile.getOriginalFilename();
    }

    @Override
    public String getFilename() {
        return this.fileName;
    }
}

@Service
public class SummaraiServiceImpl implements SummaraiService {

    private final WebClient webClient;
    private final SseServiceImpl sseService;
    private final UserDetailsServiceImpl userDetailsService;
    private final S3ServiceImpl s3Service;
    private final UserSummaryRepository userSummaryRepository;
    private final UserSummaryMapper userSummaryMapper;
    public SummaraiServiceImpl(WebClient.Builder webClientBuilder, SseServiceImpl sseService, UserDetailsServiceImpl userDetailsService, S3ServiceImpl s3Service, UserSummaryRepository userSummaryRepository, UserSummaryMapper userSummaryMapper) {
        this.webClient = webClientBuilder.baseUrl("http://localhost:8000").build();
        this.sseService = sseService;
        this.userDetailsService = userDetailsService;
        this.s3Service = s3Service;
        this.userSummaryRepository = userSummaryRepository;
        this.userSummaryMapper = userSummaryMapper;
    }

    @Override
    public void summarai(MultipartFile book, String email, int is_public,String summaryName) throws IOException {
        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("file", new MultipartFileResource(book));

        webClient.post()
                .uri("/getSummary")
                .contentType(MediaType.MULTIPART_FORM_DATA)
                .body(BodyInserters.fromMultipartData(body))
                .retrieve()
                .bodyToMono(byte[].class)
                .subscribe(response -> {
                    UserSummary userSummary = new UserSummary();
                    var temp = s3Service.uploadFile(response, summaryName);
                    String fileName = temp._1();
                    Integer pageNo = temp._2();
                    userSummary.setSummary_url(fileName);
                    userSummary.setIs_public(is_public == 1);
                    userSummary.setTitle(summaryName);
                    userSummary.setSummaryType("USER");
                    userSummary.setNormTitle(Normalizer.normalizeArabic(summaryName));
                    userSummary.setNumber_of_pages(Integer.toUnsignedLong(pageNo));
                    LocalDate localDate = LocalDate.now();  // or whatever date
                    Date nw = java.sql.Date.valueOf(localDate);
                    userSummary.setCreation_time(nw);
                    userSummaryRepository.save(userSummary);

                    SseEmitter emitter = sseService.getEmitter(email);
                    if (emitter != null) {
                        try {
                            emitter.send(SseEmitter.event().data("done"));
                            emitter.send(userSummaryMapper.toDto(userSummary));
                            emitter.complete();
//                            sseService.removeEmitter(email);
                        } catch (IOException e) {
                            emitter.completeWithError(e);
                        }
                    }
                }, error -> {
                    error.printStackTrace();
                });

        webClient.get()
                .uri("/stream")
                .accept(MediaType.TEXT_EVENT_STREAM)
                .retrieve()
                .bodyToFlux(String.class)
                .doOnNext(event -> {
                    SseEmitter emitter = sseService.getEmitter(email);
                    if (emitter != null) {
                        try {
                            emitter.send(SseEmitter.event().data(event));
                        } catch (IOException e) {
                            emitter.completeWithError(e);
                        }
                    }

                    if ("end".equalsIgnoreCase(event.trim())) {

                    }
                })
                .doOnError(Throwable::printStackTrace)
                .subscribe();
    }
    public SseEmitter createEmitter(){
        // get userid somehow
        SseEmitter emitter = new SseEmitter(0L);
        String email = userDetailsService.getCurrentUsername();
        sseService.registerEmitter(email, emitter);

        emitter.onCompletion(() -> sseService.removeEmitter(email));
        emitter.onTimeout(() -> sseService.removeEmitter(email));
        emitter.onError(e -> sseService.removeEmitter(email));

        return emitter;
    }

//    private void saveSummary(String email, int is_public,String summaryName) {
//        webClient.get()
//                .uri("/download")
//                .accept(MediaType.APPLICATION_PDF)
//                .retrieve()
//                .bodyToMono(byte[].class)
//                .doOnNext(pdfBytes -> {
//
//                    UserSummary userSummary = new UserSummary();
//                    String fileName = s3Service.uploadFile(pdfBytes, summaryName + ".pdf");
//                    userSummary.setSummary_url(fileName);
//                    userSummary.setIs_public(is_public == 1);
//                    userSummary.setTitle(summaryName);
//
//                    SseEmitter emitter = sseService.getEmitter(email);
//                    if (emitter != null) {
//                        try {
//                            emitter.send(SseEmitter.event().data("done"));
//                            emitter.complete();
//                            sseService.removeEmitter(email);
//                        } catch (IOException e) {
//                            emitter.completeWithError(e);
//                        }
//                    }
//
//                })
//                .doOnError(Throwable::printStackTrace)
//                .subscribe();
//    }
}

