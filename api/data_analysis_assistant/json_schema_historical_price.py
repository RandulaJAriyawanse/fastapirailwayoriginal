json_schema_raw = """
{
  "properties": {
    "General": {
      "Code": "",
      "Type": "",
      "Name": "",
      "Exchange": "",
      "CurrencyCode": "",
      "CurrencyName": "",
      "CurrencySymbol": "",
      "CountryName": "",
      "CountryISO": "",
      "OpenFigi": "",
      "ISIN": "",
      "LEI": "",
      "PrimaryTicker": "",
      "CIK": "",
      "EmployerIdNumber": "",
      "FiscalYearEnd": "",
      "IPODate": "",
      "InternationalDomestic": "",
      "Sector": "",
      "Industry": "",
      "GicSector": "",
      "GicGroup": "",
      "GicIndustry": "",
      "GicSubIndustry": "",
      "Description": "",
      "Address": "",
      "AddressData": {
        "Street": "",
        "City": "",
        "State": "",
        "Country": "",
        "ZIP": ""
      },
      "Listings": {
        "patternProperties": {
          "^.*$": {
            "properties": {
              "Code": "",
              "Exchange": "",
              "Name": ""
            }
          }
        }
      },
      "Officers": {
        "patternProperties": {
          "^.*$": {
            "properties": {
              "Name": "",
              "Title": "",
              "YearBorn": ""
            }
          }
        }
      },
      "Phone": "",
      "WebURL": "",
      "LogoURL": "",
      "FullTimeEmployees": "",
      "UpdatedAt": ""
    },
    "Highlights": {
      "MarketCapitalization": "",
      "MarketCapitalizationMln": "",
      "EBITDA": "",
      "PERatio": "",
      "PEGRatio": "",
      "WallStreetTargetPrice": "",
      "BookValue": "",
      "DividendShare": "",
      "DividendYield": "",
      "EarningsShare": "",
      "EPSEstimateCurrentYear": "",
      "EPSEstimateNextYear": "",
      "EPSEstimateNextQuarter": "",
      "EPSEstimateCurrentQuarter": "",
      "MostRecentQuarter": "",
      "ProfitMargin": "",
      "OperatingMarginTTM": "",
      "ReturnOnAssetsTTM": "",
      "ReturnOnEquityTTM": "",
      "RevenueTTM": "",
      "RevenuePerShareTTM": "",
      "QuarterlyRevenueGrowthYOY": "",
      "GrossProfitTTM": "",
      "DilutedEpsTTM": "",
      "QuarterlyEarningsGrowthYOY": ""
    },
    "Valuation": {
      "TrailingPE": "",
      "ForwardPE": "",
      "PriceSalesTTM": "",
      "PriceBookMRQ": "",
      "EnterpriseValue": "",
      "EnterpriseValueRevenue": "",
      "EnterpriseValueEbitda": ""
    },
    "SharesStats": {
      "SharesOutstanding": "",
      "SharesFloat": "",
      "PercentInsiders": "",
      "PercentInstitutions": "",
      "SharesShort": "",
      "SharesShortPriorMonth": "",
      "ShortRatio": "",
      "ShortPercentOutstanding": "",
      "ShortPercentFloat": ""
    },
    "Technicals": {
      "Beta": "",
      "52WeekHigh": "",
      "52WeekLow": "",
      "50DayMA": "",
      "200DayMA": "",
      "SharesShort": "",
      "SharesShortPriorMonth": "",
      "ShortRatio": "",
      "ShortPercent": ""
    },
    "SplitsDividends": {
      "properties": {
        "ForwardAnnualDividendRate": "",
        "ForwardAnnualDividendYield": "",
        "PayoutRatio": "",
        "DividendDate": "",
        "ExDividendDate": "",
        "LastSplitFactor": "",
        "LastSplitDate": "",
        "NumberDividendsByYear": {
          "patternProperties": {
            "^.*$": {
              "properties": {
                "Year": "",
                "Count": ""
              }
            }
          }
        }
      }
    },
    "Holders": {
      "properties": {
        "Funds": {
          "patternProperties": {
            "^.*$": {
              "properties": {
                "name": "",
                "date": "",
                "totalShares": "",
                "currentShares": ""
              }
            }
          }
        },
        "Institutions": {
          "$ref": "#/properties/Institutions"
        }
      }
    },
    "InsiderTransactions": "",
    "ESGScores": {
      "properties": {
        "Disclaimer": "",
        "RatingDate": "",
        "TotalEsg": "",
        "TotalEsgPercentile": "",
        "EnvironmentScore": "",
        "EnvironmentScorePercentile": "",
        "SocialScore": "",
        "SocialScorePercentile": "",
        "GovernanceScore": "",
        "GovernanceScorePercentile": "",
        "ControversyLevel": "",
        "ActivitiesInvolvement": {
          "patternProperties": {
            "^.*$": {
              "properties": {
                "Activity": "",
                "Involvement": ""
              }
            }
          }
        }
      }
    },
    "outstandingShares": {
      "properties": {
        "annual": {
          "patternProperties": {
            "^.*$": {
              "properties": {
                "date": "",
                "dateFormatted": "",
                "sharesMln": "",
                "shares": ""
              }
            }
          }
        },
        "quarterly": {
          "patternProperties": {
            "^.*$": {
              "properties": {
                "date": "",
                "dateFormatted": "",
                "sharesMln": "",
                "shares": ""
              }
            }
          }
        }
      }
    },
    "Earnings": {
      "properties": {
        "History": {
          "patternProperties": {
            "^.*$": {
              "properties": {
                "reportDate": "",
                "date": "",
                "beforeAfterMarket": "",
                "currency": "",
                "epsActual": "",
                "epsEstimate": "",
                "epsDifference": "",
                "surprisePercent": ""
              }
            }
          }
        },
        "Trend": {
          "patternProperties": {
            "date": {
              "properties": {
                "date": "",
                "period": "",
                "growth": "",
                "earningsEstimateAvg": "",
                "earningsEstimateLow": "",
                "earningsEstimateHigh": "",
                "earningsEstimateYearAgoEps": "",
                "earningsEstimateNumberOfAnalysts": "",
                "earningsEstimateGrowth": "",
                "revenueEstimateAvg": "",
                "revenueEstimateLow": "",
                "revenueEstimateHigh": "",
                "revenueEstimateYearAgoEps": "",
                "revenueEstimateNumberOfAnalysts": "",
                "revenueEstimateGrowth": "",
                "epsTrendCurrent": "",
                "epsTrend7daysAgo": "",
                "epsTrend30daysAgo": "",
                "epsTrend60daysAgo": "",
                "epsTrend90daysAgo": "",
                "epsRevisionsUpLast7days": "",
                "epsRevisionsUpLast30days": "",
                "epsRevisionsDownLast7days": "",
                "epsRevisionsDownLast30days": ""
              }
            }
          }
        },
        "Annual": {
          "patternProperties": {
            "^.*$": {
              "properties": {
                "date": "",
                "epsActual": ""
              }
            }
          }
        }
      }
    },
    "Financials": {
      "properties": {
        "Balance_Sheet": {
          "properties": {
            "currency_symbol": "",
            "quarterly": {
              "patternProperties": {
                "date": {
                  "properties": {
                    "date": "",
                    "filing_date": "",
                    "currency_symbol": "",
                    "totalAssets": "",
                    "intangibleAssets": "",
                    "earningAssets": "",
                    "otherCurrentAssets": "",
                    "totalLiab": "",
                    "totalStockholderEquity": "",
                    "deferredLongTermLiab": "",
                    "otherCurrentLiab": "",
                    "commonStock": "",
                    "capitalStock": "",
                    "retainedEarnings": "",
                    "otherLiab": "",
                    "goodWill": "",
                    "otherAssets": "",
                    "cash": "",
                    "cashAndEquivalents": "",
                    "totalCurrentLiabilities": "",
                    "currentDeferredRevenue": "",
                    "netDebt": "",
                    "shortTermDebt": "",
                    "shortLongTermDebt": "",
                    "shortLongTermDebtTotal": "",
                    "otherStockholderEquity": "",
                    "propertyPlantEquipment": "",
                    "totalCurrentAssets": "",
                    "longTermInvestments": "",
                    "netTangibleAssets": "",
                    "shortTermInvestments": "",
                    "netReceivables": "",
                    "longTermDebt": "",
                    "inventory": "",
                    "accountsPayable": "",
                    "totalPermanentEquity": "",
                    "noncontrollingInterestInConsolidatedEntity": "",
                    "temporaryEquityRedeemableNoncontrollingInterests": "",
                    "accumulatedOtherComprehensiveIncome": "",
                    "additionalPaidInCapital": "",
                    "commonStockTotalEquity": "",
                    "preferredStockTotalEquity": "",
                    "retainedEarningsTotalEquity": "",
                    "treasuryStock": "",
                    "accumulatedAmortization": "",
                    "nonCurrrentAssetsOther": "",
                    "deferredLongTermAssetCharges": "",
                    "nonCurrentAssetsTotal": "",
                    "capitalLeaseObligations": "",
                    "longTermDebtTotal": "",
                    "nonCurrentLiabilitiesOther": "",
                    "nonCurrentLiabilitiesTotal": "",
                    "negativeGoodwill": "",
                    "warrants": "",
                    "preferredStockRedeemable": "",
                    "capitalSurpluse": "",
                    "liabilitiesAndStockholdersEquity": "",
                    "cashAndShortTermInvestments": "",
                    "propertyPlantAndEquipmentGross": "",
                    "propertyPlantAndEquipmentNet": "",
                    "accumulatedDepreciation": "",
                    "netWorkingCapital": "",
                    "netInvestedCapital": "",
                    "commonStockSharesOutstanding": ""
                  }
                }
              }
            },
            "yearly": {
              "patternProperties": {
                "date": {
                  "properties": {
                    "date": "",
                    "filing_date": "",
                    "currency_symbol": "",
                    "totalAssets": "",
                    "intangibleAssets": "",
                    "earningAssets": "",
                    "otherCurrentAssets": "",
                    "totalLiab": "",
                    "totalStockholderEquity": "",
                    "deferredLongTermLiab": "",
                    "otherCurrentLiab": "",
                    "commonStock": "",
                    "capitalStock": "",
                    "retainedEarnings": "",
                    "otherLiab": "",
                    "goodWill": "",
                    "otherAssets": "",
                    "cash": "",
                    "cashAndEquivalents": "",
                    "totalCurrentLiabilities": "",
                    "currentDeferredRevenue": "",
                    "netDebt": "",
                    "shortTermDebt": "",
                    "shortLongTermDebt": "",
                    "shortLongTermDebtTotal": "",
                    "otherStockholderEquity": "",
                    "propertyPlantEquipment": "",
                    "totalCurrentAssets": "",
                    "longTermInvestments": "",
                    "netTangibleAssets": "",
                    "shortTermInvestments": "",
                    "netReceivables": "",
                    "longTermDebt": "",
                    "inventory": "",
                    "accountsPayable": "",
                    "totalPermanentEquity": "",
                    "noncontrollingInterestInConsolidatedEntity": "",
                    "temporaryEquityRedeemableNoncontrollingInterests": "",
                    "accumulatedOtherComprehensiveIncome": "",
                    "additionalPaidInCapital": "",
                    "commonStockTotalEquity": "",
                    "preferredStockTotalEquity": "",
                    "retainedEarningsTotalEquity": "",
                    "treasuryStock": "",
                    "accumulatedAmortization": "",
                    "nonCurrrentAssetsOther": "",
                    "deferredLongTermAssetCharges": "",
                    "nonCurrentAssetsTotal": "",
                    "capitalLeaseObligations": "",
                    "longTermDebtTotal": "",
                    "nonCurrentLiabilitiesOther": "",
                    "nonCurrentLiabilitiesTotal": "",
                    "negativeGoodwill": "",
                    "warrants": "",
                    "preferredStockRedeemable": "",
                    "capitalSurpluse": "",
                    "liabilitiesAndStockholdersEquity": "",
                    "cashAndShortTermInvestments": "",
                    "propertyPlantAndEquipmentGross": "",
                    "propertyPlantAndEquipmentNet": "",
                    "accumulatedDepreciation": "",
                    "netWorkingCapital": "",
                    "netInvestedCapital": "",
                    "commonStockSharesOutstanding": ""
                  }
                }
              }
            }
          }
        },
        "Cash_Flow": {
          "properties": {
            "currency_symbol": "",
            "quarterly": {
              "patternProperties": {
                "date": {
                  "properties": {
                    "date": "",
                    "filing_date": "",
                    "currency_symbol": "",
                    "investments": "",
                    "changeToLiabilities": "",
                    "totalCashflowsFromInvestingActivities": "",
                    "netBorrowings": "",
                    "totalCashFromFinancingActivities": "",
                    "changeToOperatingActivities": "",
                    "netIncome": "",
                    "changeInCash": "",
                    "beginPeriodCashFlow": "",
                    "endPeriodCashFlow": "",
                    "totalCashFromOperatingActivities": "",
                    "issuanceOfCapitalStock": "",
                    "depreciation": "",
                    "otherCashflowsFromInvestingActivities": "",
                    "dividendsPaid": "",
                    "changeToInventory": "",
                    "changeToAccountReceivables": "",
                    "salePurchaseOfStock": "",
                    "otherCashflowsFromFinancingActivities": "",
                    "changeToNetincome": "",
                    "capitalExpenditures": "",
                    "changeReceivables": "",
                    "cashFlowsOtherOperating": "",
                    "exchangeRateChanges": "",
                    "cashAndCashEquivalentsChanges": "",
                    "changeInWorkingCapital": "",
                    "stockBasedCompensation": "",
                    "otherNonCashItems": "",
                    "freeCashFlow": ""
                  }
                }
              }
            },
            "yearly": {
              "patternProperties": {
                "date": {
                  "properties": {
                    "date": "",
                    "filing_date": "",
                    "currency_symbol": "",
                    "investments": "",
                    "changeToLiabilities": "",
                    "totalCashflowsFromInvestingActivities": "",
                    "netBorrowings": "",
                    "totalCashFromFinancingActivities": "",
                    "changeToOperatingActivities": "",
                    "netIncome": "",
                    "changeInCash": "",
                    "beginPeriodCashFlow": "",
                    "endPeriodCashFlow": "",
                    "totalCashFromOperatingActivities": "",
                    "issuanceOfCapitalStock": "",
                    "depreciation": "",
                    "otherCashflowsFromInvestingActivities": "",
                    "dividendsPaid": "",
                    "changeToInventory": "",
                    "changeToAccountReceivables": "",
                    "salePurchaseOfStock": "",
                    "otherCashflowsFromFinancingActivities": "",
                    "changeToNetincome": "",
                    "capitalExpenditures": "",
                    "changeReceivables": "",
                    "cashFlowsOtherOperating": "",
                    "exchangeRateChanges": "",
                    "cashAndCashEquivalentsChanges": "",
                    "changeInWorkingCapital": "",
                    "stockBasedCompensation": "",
                    "otherNonCashItems": "",
                    "freeCashFlow": ""
                  }
                }
              }
            }
          }
        },
        "Income_Statement": {
          "properties": {
            "currency_symbol": "",
            "quarterly": {
              "patternProperties": {
                "date": {
                  "properties": {
                    "date": "",
                    "filing_date": "",
                    "currency_symbol": "",
                    "researchDevelopment": "",
                    "effectOfAccountingCharges": "",
                    "incomeBeforeTax": "",
                    "minorityInterest": "",
                    "netIncome": "",
                    "sellingGeneralAdministrative": "",
                    "sellingAndMarketingExpenses": "",
                    "grossProfit": "",
                    "reconciledDepreciation": "",
                    "ebit": "",
                    "ebitda": "",
                    "depreciationAndAmortization": "",
                    "nonOperatingIncomeNetOther": "",
                    "operatingIncome": "",
                    "otherOperatingExpenses": "",
                    "interestExpense": "",
                    "taxProvision": "",
                    "interestIncome": "",
                    "netInterestIncome": "",
                    "extraordinaryItems": "",
                    "nonRecurring": "",
                    "otherItems": "",
                    "incomeTaxExpense": "",
                    "totalRevenue": "",
                    "totalOperatingExpenses": "",
                    "costOfRevenue": "",
                    "totalOtherIncomeExpenseNet": "",
                    "discontinuedOperations": "",
                    "netIncomeFromContinuingOps": "",
                    "netIncomeApplicableToCommonShares": "",
                    "preferredStockAndOtherAdjustments": ""
                  }
                }
              }
            },
            "yearly": {
              "patternProperties": {
                "date": {
                  "properties": {
                    "date": "",
                    "filing_date": "",
                    "currency_symbol": "",
                    "researchDevelopment": "",
                    "effectOfAccountingCharges": "",
                    "incomeBeforeTax": "",
                    "minorityInterest": "",
                    "netIncome": "",
                    "sellingGeneralAdministrative": "",
                    "sellingAndMarketingExpenses": "",
                    "grossProfit": "",
                    "reconciledDepreciation": "",
                    "ebit": "",
                    "ebitda": "",
                    "depreciationAndAmortization": "",
                    "nonOperatingIncomeNetOther": "",
                    "operatingIncome": "",
                    "otherOperatingExpenses": "",
                    "interestExpense": "",
                    "taxProvision": "",
                    "interestIncome": "",
                    "netInterestIncome": "",
                    "extraordinaryItems": "",
                    "nonRecurring": "",
                    "otherItems": "",
                    "incomeTaxExpense": "",
                    "totalRevenue": "",
                    "totalOperatingExpenses": "",
                    "costOfRevenue": "",
                    "totalOtherIncomeExpenseNet": "",
                    "discontinuedOperations": "",
                    "netIncomeFromContinuingOps": "",
                    "netIncomeApplicableToCommonShares": "",
                    "preferredStockAndOtherAdjustments": ""
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
"""
