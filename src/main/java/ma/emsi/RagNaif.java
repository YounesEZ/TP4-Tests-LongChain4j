package ma.emsi;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;


public class RagNaif {
    public static void main(String[] args) {

        ChatLanguageModel modele = GoogleAiGeminiChatModel.builder()
                .apiKey(System.getenv("GEMINI_KEY"))
                .modelName("gemini-1.5-flash")
                .temperature(0.7)
                .build();

        Path pathRessource;
        try {
            String cheminRessource = "/ml.pdf"; // chemin absolu
            // MaClass désigne, par exemple, la classe qui contient ce code.
            URL fileUrl = RagNaif.class.getResource(cheminRessource);
            pathRessource = Paths.get(fileUrl.toURI());
        } catch (URISyntaxException e) {
            throw new RuntimeException(e); // ou un autre traitement du problème...
        }

        DocumentParser documentParser = new ApacheTikaDocumentParser();

        Document document = FileSystemDocumentLoader.loadDocument(pathRessource, documentParser);

        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);

        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        EmbeddingStore embeddingStore = new InMemoryEmbeddingStore();

        EmbeddingStoreIngestor ingestor = EmbeddingStoreIngestor.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .documentSplitter(splitter)
                .build();
        ingestor.ingest(document);

        ContentRetriever retriever = EmbeddingStoreContentRetriever.builder()
                .embeddingModel(embeddingModel)
                .embeddingStore(embeddingStore)
                .maxResults(5)
                .minScore(0.5)
                .build();

        //ChatMemory chatMemory = MessageWindowChatMemory.withMaxMessages(10); // n'est pas utilisé pour ce test

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatLanguageModel(modele)
                .contentRetriever(retriever)
                .build();
        //Etape4
        String reponse = assistant.chat("Cest quoi l'objectif du cours de Richard Grin?");
        System.out.println(reponse);

    }
}
