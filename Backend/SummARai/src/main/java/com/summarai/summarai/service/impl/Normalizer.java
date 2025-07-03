package com.summarai.summarai.service.impl;

public class Normalizer {
    public static String normalizeArabic(String input) {
        if (input == null) return null;

        // Remove Arabic diacritics (harakat)
        String cleaned = input.replaceAll("[\\u064B-\\u0652\\u0670]", "");

        // Normalize common hamza-related forms
        cleaned = cleaned
                .replace("أ", "ا")
                .replace("إ", "ا")
                .replace("آ", "ا")
                .replace("ؤ", "و")
                .replace("ئ", "ي");

        return cleaned;
    }
}
