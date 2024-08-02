`%>%` <- dplyr::`%>%`
library(AnnotationDbi)
# custom functions
source(file.path("util", "plier_util.R"))


obtain_b_matrix <- function(data_path, b_matrix_path) {
    nares.data <- readr::read_csv(file.path(data_path),
                              progress = FALSE)

    symbol.obj <- org.Hs.eg.db::org.Hs.egSYMBOL
    mapped.genes <- AnnotationDbi::mappedkeys(symbol.obj)
    symbol.list <- as.list(symbol.obj[mapped.genes])
    symbol.df <- as.data.frame(cbind(names(symbol.list), unlist(symbol.list)))
    colnames(symbol.df) <- c("EntrezID", "GeneSymbol")
    colnames(nares.data)[1] <- "EntrezID"


    symbol.df$EntrezID <- as.integer(as.character(symbol.df$EntrezID))
    annot.nares.data <- dplyr::inner_join(symbol.df, nares.data, by = "EntrezID")


    exprs.mat <- dplyr::select(annot.nares.data, -EntrezID)
    rownames(exprs.mat) <- exprs.mat$GeneSymbol
    exprs.mat <- as.matrix(dplyr::select(exprs.mat, -GeneSymbol))


    plier.results <- readRDS(file.path("data", "expression_data", "recount2", "recount2_PLIER_data", 
                                   "recount_PLIER_model.RDS"))


    iso.b.matrix <- GetNewDataB(exprs.mat = exprs.mat,
                            plier.model = plier.results)

    write.csv(iso.b.matrix, b_matrix_path)
    
    # return (5 * x)
}



obtain_b_matrix('coder_data/all_transcriptomics_mp.csv', 'b_matrices/all_transcriptomics_b.csv')
obtain_b_matrix('coder_data/all_proteomics_mp.csv', 'b_matrices/all_proteomics_b.csv')
obtain_b_matrix('coder_data/all_copy_number_mp.csv', 'b_matrices/all_copy_number_b.csv')