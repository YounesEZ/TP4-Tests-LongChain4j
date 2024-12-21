package ma.emsi;


import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.rag.query.router.DefaultQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.rag.query.transformer.CompressingQueryTransformer;
import dev.langchain4j.rag.query.transformer.QueryTransformer;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.web.search.WebSearchEngine;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;

import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;


public class RagNaif6 {

    private static void configureLogger() {
        // Configure le logger sous-jacent (java.util.logging)
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE); // Ajuster niveau
        // Ajouter un handler pour la console pour faire afficher les logs
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.addHandler(handler);
    }

    public static void main(String[] args) {

        configureLogger();

        ChatLanguageModel modele = GoogleAiGeminiChatModel.builder()
                .apiKey(System.getenv("GEMINI_KEY"))
                .modelName("gemini-1.5-flash")
                .temperature(0.7)
                .logRequestsAndResponses(true)
                .build();


        Path pathRessource2;
        try {

            String cheminRessource2 = "/rag.pdf"; //

            URL fileUrl = RagNaif.class.getResource(cheminRessource2);
            pathRessource2 = Paths.get(fileUrl.toURI());

        } catch (URISyntaxException e) {
            throw new RuntimeException(e); // ou un autre traitement du problème...
        }


        DocumentParser documentParser = new ApacheTikaDocumentParser();

        Document document2 = FileSystemDocumentLoader.loadDocument(pathRessource2, documentParser);

        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);

        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();


        EmbeddingStore embeddingStore2 = new InMemoryEmbeddingStore();


        EmbeddingStoreIngestor ingestor2 = EmbeddingStoreIngestor.builder()
                .embeddingStore(embeddingStore2)
                .embeddingModel(embeddingModel)
                .documentSplitter(splitter)
                .build();

        ingestor2.ingest(document2);


        ContentRetriever retriever2 = EmbeddingStoreContentRetriever.builder()
                .embeddingModel(embeddingModel)
                .embeddingStore(embeddingStore2)
                .maxResults(2)
                .minScore(0.5)
                .build();


        WebSearchEngine webSearchEngine = TavilyWebSearchEngine.builder()
                .apiKey(System.getenv("TAVILY_KEY"))
                .build();

        ContentRetriever webretriever = WebSearchContentRetriever.builder()
                .webSearchEngine(webSearchEngine)
                .build();


        QueryRouter routerWebOrPDF = new DefaultQueryRouter(List.of(retriever2, webretriever));

        QueryTransformer transformer = CompressingQueryTransformer.builder()
                .chatLanguageModel(modele)
                .build();

        RetrievalAugmentor augmentor = DefaultRetrievalAugmentor.builder()
                .queryTransformer(transformer)
                .queryRouter(routerWebOrPDF)
                .build();


        ChatMemory chatMemory = MessageWindowChatMemory.withMaxMessages(10);


        Assistant assistant = AiServices.builder(Assistant.class)
                .chatLanguageModel(modele)
                .chatMemory(chatMemory)
                .retrievalAugmentor(augmentor)
                .build();


        try (Scanner scanner = new Scanner(System.in)) {
            while (true) {
                System.out.println("==================================================");
                System.out.println("Posez votre question : ");
                String question = scanner.nextLine();
                System.out.println("==================================================");
                if ("fin".equalsIgnoreCase(question)) {
                    break;
                }
                String reponse = assistant.chat(question);
                System.out.println("==================================================");
                System.out.println("Assistant : " + reponse);
            }
        }
    }
}
